#!/home/lms/miniforge3/envs/foundationpose/bin/python3
"""
NVIDIA FoundationPose Persistent Inference Server.

Runs in the `foundationpose` Conda environment (PyTorch 2.6 + CUDA 12.4 + nvdiffrast + pytorch3d)
and serves 6D pose estimation requests over localhost TCP (127.0.0.1:5555) for the ROS 2
perception_pick_and_place node.
"""

import io
import json
import logging
import os
import pickle
import socket
import struct
import sys

FOUNDATIONPOSE_DIR = "/home/lms/FoundationPose"
if FOUNDATIONPOSE_DIR not in sys.path:
    sys.path.insert(0, FOUNDATIONPOSE_DIR)

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

from estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from Utils import set_seed

HOST = "127.0.0.1"
PORT = 5555


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(conn: socket.socket) -> dict:
    raw_len = recv_exact(conn, 4)
    (msg_len,) = struct.unpack(">I", raw_len)
    payload = recv_exact(conn, msg_len)
    return pickle.loads(payload)


def send_msg(conn: socket.socket, obj: dict):
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack(">I", len(payload)) + payload)


class MultiPartFoundationPoseEngine:
    """Holds shared Scorer/Refiner/Rasterizer and per-part FoundationPose estimators."""

    def __init__(self, manifest_path: str):
        set_seed(0)
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(manifest_path)))
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        print("[FoundationPoseServer] Loading ScorePredictor & PoseRefinePredictor onto GPU...")
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.estimators = {}
        self.tf_to_center = {}

        for item in self.manifest:
            part_id = item["id"]
            mesh_path = os.path.join(self.base_dir, item["mesh_obj"])
            mesh = trimesh.load(mesh_path)
            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=self.scorer,
                refiner=self.refiner,
                debug_dir="/tmp/fp_debug",
                debug=0,
                glctx=self.glctx,
            )
            self.estimators[part_id] = est
            self.tf_to_center[part_id] = est.get_tf_to_centered_mesh().detach().cpu().numpy()
            print(f"[FoundationPoseServer] Initialized estimator for {part_id} (diameter={est.diameter:.4f}m)")

        print("[FoundationPoseServer] All 6 CAD estimators ready.")

    def estimate_batch(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, masks: dict, iteration: int = 5) -> dict:
        results = {}
        for part_id, mask in masks.items():
            if part_id not in self.estimators:
                continue
            mask_bool = np.asarray(mask, dtype=bool)
            if mask_bool.sum() < 25:
                print(f"[FoundationPoseServer] Skipping {part_id}: mask too small ({mask_bool.sum()} px)")
                continue

            est = self.estimators[part_id]
            try:
                pose_uncentered = est.register(
                    K=K.astype(np.float64),
                    rgb=rgb,
                    depth=depth.astype(np.float64),
                    ob_mask=mask_bool,
                    iteration=iteration,
                )
                pose_uncentered = np.asarray(pose_uncentered, dtype=np.float64)
                # est.pose_last is the centered bounding-box pose (invariant to cube symmetries)
                pose_centered = est.pose_last.detach().cpu().numpy().astype(np.float64)
                if pose_centered.ndim == 3:
                    pose_centered = pose_centered[0]
                results[part_id] = {
                    "pose_centered_in_cam": pose_centered,
                    "pose_bottom_in_cam": pose_uncentered,
                }
                print(
                    f"[FoundationPoseServer] {part_id}: cam_pos=({pose_centered[0,3]:.3f}, "
                    f"{pose_centered[1,3]:.3f}, {pose_centered[2,3]:.3f})",
                    flush=True,
                )
            except Exception as e:
                import traceback
                print(f"[FoundationPoseServer] Error estimating {part_id}: {e}", flush=True)
                traceback.print_exc()
        return results


def main():
    logging.getLogger().setLevel(logging.WARNING)
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "parts_manifest.json",
    )

    os.makedirs("/tmp/fp_debug", exist_ok=True)
    engine = MultiPartFoundationPoseEngine(manifest_path)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(4)
    print(f"[FoundationPoseServer] Listening on {HOST}:{PORT}...")
    sys.stdout.flush()

    while True:
        conn, addr = srv.accept()
        try:
            req = recv_msg(conn)
            cmd = req.get("cmd", "estimate")
            if cmd == "ping":
                send_msg(conn, {"status": "ok"})
            elif cmd == "shutdown":
                send_msg(conn, {"status": "bye"})
                conn.close()
                break
            elif cmd == "estimate":
                res = engine.estimate_batch(
                    rgb=req["rgb"],
                    depth=req["depth"],
                    K=req["K"],
                    masks=req["masks"],
                    iteration=req.get("iteration", 5),
                )
                send_msg(conn, {"status": "ok", "results": res})
            else:
                send_msg(conn, {"status": "error", "error": f"Unknown cmd: {cmd}"})
        except Exception as e:
            try:
                send_msg(conn, {"status": "error", "error": str(e)})
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
