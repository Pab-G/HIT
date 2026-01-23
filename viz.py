import os

import numpy as np
import trimesh


def visualize_sampling(bone_path, num_samples=5000):
    if not os.path.exists(bone_path):
        print(f"File not found: {bone_path}")
        return

    mesh = trimesh.load(bone_path)

    # --- METHOD 2: Barycentric Hull Sampling (No rtree needed) ---
    print("Sampling Red points (Hull Volume)...")
    hull = mesh.convex_hull
    # We create tetrahedrons between the center and each face to fill the volume
    v3 = hull.center_mass
    faces = hull.faces
    vertices = hull.vertices

    hull_pts = []
    for _ in range(num_samples):
        face = faces[np.random.randint(len(faces))]
        v0, v1, v2 = vertices[face]
        w = np.random.rand(4)
        w /= w.sum()
        hull_pts.append(w[0] * v0 + w[1] * v1 + w[2] * v2 + w[3] * v3)

    pc_hull = trimesh.points.PointCloud(hull_pts, colors=[255, 0, 0, 255])

    # --- METHOD 3: Jitter Sampling ---
    print("Sampling Blue points (Surface Jitter)...")
    surface_points = mesh.vertices
    idx = np.random.choice(len(surface_points), num_samples)
    jitter_pts = surface_points[idx] + np.random.normal(0, 0.008, (num_samples, 3))

    pc_jitter = trimesh.points.PointCloud(jitter_pts, colors=[0, 0, 255, 255])

    # --- TRANSLATION (Offset for better viewing) ---
    # Move the jitter points to the right so they don't overlap with the hull points
    pc_jitter.apply_translation([0.2, 0, 0])

    # Create a second copy of the mesh for the offset points
    mesh_offset = mesh.copy()
    mesh_offset.apply_translation([0.2, 0, 0])

    # Visual settings
    mesh.visual.face_colors = [200, 200, 200, 50]  # Very translucent
    mesh_offset.visual.face_colors = [200, 200, 200, 50]

    print("Opening viewer: Left = Hull (Red) | Right = Jitter (Blue)")

    # ADD BOTH TO THE SCENE
    scene = trimesh.Scene([mesh, pc_hull, mesh_offset, pc_jitter])
    scene.show()


def meg(bone_path, num_samples=3000):
    if not os.path.exists(bone_path):
        return

    data = trimesh.load(bone_path)
    surface_pts = data.vertices

    # --- ROBUST SAMPLING ---
    # Pick indices safely regardless of bone size
    pop_size = len(surface_pts)
    use_replace = pop_size < num_samples
    idx = np.random.choice(pop_size, num_samples, replace=use_replace)
    base_pts = surface_pts[idx]

    # --- THE THICKNESS LOGIC ---
    center = np.mean(base_pts, axis=0)
    direction = center - base_pts
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8

    # Layer 1 (Surface), Layer 2 (3mm deep), Layer 3 (7mm deep)
    # We use multiple layers to help the specialist model understand 'interior'
    l1 = base_pts
    l2 = base_pts + (direction * 0.003)
    l3 = base_pts + (direction * 0.007)

    # Visualization
    pc_surface = trimesh.points.PointCloud(l1, colors=[255, 0, 0, 255])
    internal_pts = np.concatenate([l2, l3], axis=0)
    pc_internal = trimesh.points.PointCloud(internal_pts, colors=[0, 0, 255, 255])

    # Offset for side-by-side view
    pc_internal.apply_translation([0.2, 0, 0])

    print(f"Showing {os.path.basename(bone_path)}")
    print("LEFT: Original Surface | RIGHT: Specialist Volume (Synthesized)")
    trimesh.Scene([pc_surface, pc_internal]).show()


if __name__ == "__main__":
    # Ensure this path is correct
    TEST_PATH = "mri_bones_release_v2/train/male/BH1470/per_part_pc/Femur_pc.ply"
    visualize_sampling(TEST_PATH)
