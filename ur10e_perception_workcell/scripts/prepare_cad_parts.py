#!/usr/bin/env python3
"""Prepare scaled CAD models, Gazebo SDFs, FoundationPose OBJ meshes, and 3D UI thumbnails."""

import json
import os
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FREE_CAD_DIR = os.path.join(BASE_DIR, "models", "FREE_CAD_PARTS")
MODELS_DIR = os.path.join(BASE_DIR, "models")
THUMBNAILS_DIR = os.path.join(MODELS_DIR, "part_thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

# 2 rows x 3 columns grid on the table (top surface at Z = 0.800m)
# Row 0 (front, closer to robot): X = 0.72m
# Row 1 (back, further row):      X = 0.96m
# Cols: Left Y = +0.22m, Center Y = 0.00m, Right Y = -0.22m
PARTS_SPEC = [
    {
        "id": "part_bracket_small",
        "display_name": "Bracket 23-040-6",
        "stl_rel": "196149511-21-23-040-6/23-040-6.stl",
        "scale": 0.003,
        "euler_deg": [0, 0, 0],
        "row": 0,
        "col": 0,
        "x": 0.72,
        "y": 0.22,
        "rgba": [0.98, 0.45, 0.05, 1.0],  # Bright Orange
        "hsv_ranges": [[[11, 100, 80], [20, 255, 255]]],
    },
    {
        "id": "red_cube",
        "display_name": "Red Cube",
        "stl_rel": None,
        "box_size": [0.05, 0.05, 0.05],
        "scale": 1.0,
        "euler_deg": [0, 0, 0],
        "row": 0,
        "col": 1,
        "x": 0.72,
        "y": 0.00,
        "rgba": [0.90, 0.10, 0.10, 1.0],  # Crimson Red
        "hsv_ranges": [[[0, 90, 60], [10, 255, 255]], [[170, 90, 60], [180, 255, 255]]],
    },
    {
        "id": "part_lindapter",
        "display_name": "Lindapter",
        "stl_rel": "lindapter/9531410.stl",
        "scale": 0.002,
        "euler_deg": [0, 0, 0],
        "row": 0,
        "col": 2,
        "x": 0.72,
        "y": -0.22,
        "rgba": [0.15, 0.75, 0.25, 1.0],  # Emerald Green
        "hsv_ranges": [[[45, 80, 60], [85, 255, 255]]],
    },
    {
        "id": "part_clamping_bar",
        "display_name": "Clamping Bar",
        "stl_rel": "positioning_clamping_bar/halderv_eh_1149_000.stl",
        "scale": 0.0011,
        "euler_deg": [0, 0, 0],
        "row": 1,
        "col": 0,
        "x": 0.96,
        "y": 0.22,
        "rgba": [0.95, 0.88, 0.10, 1.0],  # Bright Yellow
        "hsv_ranges": [[[23, 100, 80], [36, 255, 255]]],
    },
    {
        "id": "part_square_nut",
        "display_name": "Square Nut",
        "stl_rel": "square_nut/2454852.stl",
        "scale": 0.0045,
        "euler_deg": [0, 0, 0],
        "row": 1,
        "col": 1,
        "x": 0.96,
        "y": 0.00,
        "rgba": [0.65, 0.20, 0.85, 1.0],  # Violet/Purple
        "hsv_ranges": [[[132, 70, 60], [165, 255, 255]]],
    },
    {
        "id": "part_t_connector",
        "display_name": "T-Connector",
        "stl_rel": "t_connector/7982053.stl",
        "scale": 0.0015,
        "euler_deg": [0, 90, 0],  # Lay flat on its 67.5x67.5mm face
        "row": 1,
        "col": 2,
        "x": 0.96,
        "y": -0.22,
        "rgba": [0.10, 0.78, 0.85, 1.0],  # Cyan/Teal
        "hsv_ranges": [[[86, 80, 60], [99, 255, 255]]],
    },
]


def render_thumbnail(mesh: trimesh.Trimesh, rgba: list, title: str, out_path: str):
    """Render a clean isometric 3D PNG thumbnail of the CAD mesh."""
    fig = plt.figure(figsize=(2.4, 2.4), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#1e222a")
    ax.set_facecolor("#1e222a")

    # Subsample faces if too dense for fast crisp rendering
    faces = mesh.triangles
    if len(faces) > 2500:
        idx = np.linspace(0, len(faces) - 1, 2500, dtype=int)
        faces = faces[idx]

    # Simple directional shading for 3D depth perception
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.maximum(norms, 1e-9))
    light_dir = np.array([0.4, -0.5, 0.75])
    light_dir /= np.linalg.norm(light_dir)
    intensity = np.clip(0.45 + 0.55 * np.abs(normals @ light_dir), 0.3, 1.0)

    base_rgb = np.array(rgba[:3])
    face_colors = np.clip(intensity[:, None] * base_rgb[None, :], 0.0, 1.0)
    face_rgba = np.column_stack([face_colors, np.ones(len(face_colors))])

    poly = Poly3DCollection(faces, facecolors=face_rgba, edgecolors=(0, 0, 0, 0.08), linewidths=0.2)
    ax.add_collection3d(poly)

    # Equal aspect ratio centered on mesh
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    max_range = np.max(bounds[1] - bounds[0]) * 0.62
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.view_init(elev=28, azim=-48)
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_model_sdf(part_id: str, rgba: list, extents: list, out_path: str):
    """Write Gazebo SDF file for the CAD part."""
    r, g, b, a = rgba
    dx, dy, dz = extents
    mass = 0.15
    ixx = (1.0 / 12.0) * mass * (dy * dy + dz * dz)
    iyy = (1.0 / 12.0) * mass * (dx * dx + dz * dz)
    izz = (1.0 / 12.0) * mass * (dx * dx + dy * dy)
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{part_id}">
    <static>false</static>
    <link name="part_link">
      <inertial>
        <pose>0 0 {dz / 2.0:.5f} 0 0 0</pose>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx:.7f}</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>{iyy:.7f}</iyy>
          <iyz>0</iyz>
          <izz>{izz:.7f}</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <mesh>
            <uri>package://ur10e_perception_workcell/models/{part_id}/meshes/{part_id}.stl</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>1.0</mu>
              <mu2>1.0</mu2>
            </ode>
          </friction>
          <contact>
            <ode>
              <kp>1000000.0</kp>
              <kd>100.0</kd>
            </ode>
          </contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>package://ur10e_perception_workcell/models/{part_id}/meshes/{part_id}.stl</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
        <material>
          <ambient>{r * 0.6:.3f} {g * 0.6:.3f} {b * 0.6:.3f} {a:.2f}</ambient>
          <diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.2f}</diffuse>
          <specular>0.3 0.3 0.3 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    with open(out_path, "w") as f:
        f.write(sdf)


def main():
    manifest = []
    for spec in PARTS_SPEC:
        part_id = spec["id"]
        if spec.get("stl_rel"):
            src_stl = os.path.join(FREE_CAD_DIR, spec["stl_rel"])
            mesh = trimesh.load(src_stl)
        else:
            mesh = trimesh.creation.box(extents=spec["box_size"])

        # Apply scale
        mesh.apply_scale(spec["scale"])

        # Apply stabilizing orientation if specified
        if any(spec["euler_deg"]):
            rot = trimesh.transformations.euler_matrix(
                np.radians(spec["euler_deg"][0]),
                np.radians(spec["euler_deg"][1]),
                np.radians(spec["euler_deg"][2]),
            )
            mesh.apply_transform(rot)

        # Center XY around 0 and place bottom exactly at Z = 0.0
        bounds = mesh.bounds
        cx = (bounds[0][0] + bounds[1][0]) / 2.0
        cy = (bounds[0][1] + bounds[1][1]) / 2.0
        z_min = bounds[0][2]
        mesh.apply_translation([-cx, -cy, -z_min])

        extents = mesh.extents.tolist()
        part_dir = os.path.join(MODELS_DIR, part_id)
        mesh_dir = os.path.join(part_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        out_stl = os.path.join(mesh_dir, f"{part_id}.stl")
        out_obj = os.path.join(mesh_dir, f"{part_id}.obj")
        mesh.export(out_stl)

        # Bake vertex colors into OBJ for FoundationPose nvdiffrast rendering
        rgba_u8 = [int(round(c * 255)) for c in spec["rgba"]]
        mesh.visual.vertex_colors = np.tile(rgba_u8, (len(mesh.vertices), 1)).astype(np.uint8)
        mesh.export(out_obj)

        out_sdf = os.path.join(part_dir, "model.sdf")
        write_model_sdf(part_id, spec["rgba"], extents, out_sdf)

        out_thumb = os.path.join(THUMBNAILS_DIR, f"{part_id}.png")
        render_thumbnail(mesh, spec["rgba"], spec["display_name"], out_thumb)

        print(f"Prepared {part_id}: extents={[round(e, 4) for e in extents]}, grid=({spec['row']},{spec['col']}) @ ({spec['x']}, {spec['y']})")

        manifest.append({
            "id": part_id,
            "display_name": spec["display_name"],
            "row": spec["row"],
            "col": spec["col"],
            "x": spec["x"],
            "y": spec["y"],
            "z": 0.800,
            "extents": extents,
            "rgba": spec["rgba"],
            "hsv_ranges": spec["hsv_ranges"],
            "mesh_stl": f"models/{part_id}/meshes/{part_id}.stl",
            "mesh_obj": f"models/{part_id}/meshes/{part_id}.obj",
            "thumbnail": f"models/part_thumbnails/{part_id}.png",
        })

    manifest_path = os.path.join(MODELS_DIR, "parts_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
