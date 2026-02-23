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

import pickle as pkl

import pyrender
import torch

import hit.hit_config as cg
from hit.model.mysmpl import MySmpl


def compare_smpl_datasets(path_specialist, path_repackaged, subject_idx=11, gender='female'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize SMPL model
    print(f"--- Loading SMPL ({gender}) ---")
    smpl = MySmpl(model_path=cg.smplx_models_path, gender=gender).to(device)
    
    all_verts = []
    meshes = []
    colors = [[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.5]] # Red (Spec), Blue (Orig)

    # 2. Load Data
    with open(path_specialist, 'rb') as f:
        data_spec = pkl.load(f)
    with open(path_repackaged, 'rb') as f:
        full_orig = pkl.load(f)
    # Check if we have the index in the repackaged data
    total_subjects = len(full_orig["seq_names"])
    if subject_idx >= total_subjects:
        raise ValueError(f"subject_idx {subject_idx} is out of bounds (Total: {total_subjects})")

    print(f"Comparing Specialist file vs Original subject: {full_orig['seq_names'][subject_idx]}")

    # 3. Process both entries
    for i in range(2):
        if i == 0:
            name = "Specialist (MRI Bones)"
            # Specialist data is usually a flat dict for one person
            raw_betas = data_spec.get('betas', np.zeros(10))
        else:
            name = f"Original (Index {subject_idx})"
            # Original data is columnar, extract index from the list
            raw_betas = full_orig['betas'][subject_idx]

        # --- SHAPE NORMALIZATION ---
        if torch.is_tensor(raw_betas):
            raw_betas = raw_betas.detach().cpu().numpy()
        
        # Force to (1, 10) for SMPL einsum compatibility
        betas_np = np.array(raw_betas).flatten()[:10].astype(np.float32)
        betas_torch = torch.from_numpy(betas_np).unsqueeze(0).to(device)
        
        # Zero out pose/transl to get the Canonical T-Pose alignment
        empty_pose = torch.zeros((1, 69), dtype=torch.float32).to(device)
        empty_transl = torch.zeros((1, 3), dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output = smpl(betas=betas_torch, body_pose=empty_pose, transl=empty_transl)
            verts = output.vertices[0].detach().cpu().numpy()
            all_verts.append(verts)
        
        mesh = trimesh.Trimesh(vertices=verts, faces=smpl.faces, process=False)
        mesh.visual.vertex_colors = colors[i]
        meshes.append(mesh)

        print(f"Target {i+1}: {name}")
        print(f"  - Betas: {betas_np[:3]}...")
        print(f"  - Height: {verts[:,1].max() - verts[:,1].min():.4f}m")

    # --- CALCULATE GEOMETRIC DRIFT ---
    drift_vec = np.linalg.norm(all_verts[0] - all_verts[1], axis=1)
    avg_drift = np.mean(drift_vec)
    
    print("\n📏 RESULTS:")
    print(f"  - Average Vertex Drift: {avg_drift*1000:.4f} mm")
    
    if avg_drift > 0.001: # Threshold of 1mm
        print("⚠️  MISALIGNMENT DETECTED: The unposing coordinates will differ.")
    else:
        print("✅ ALIGNMENT PERFECT: The templates are geometrically identical.")

    # 4. Render
    scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
    for m in meshes:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    
    print("\n--- Visualizing Override: Red (Spec) vs Blue (Orig) ---")
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":
    # Ensure this path is correct
    # Usage
    path_a = "/home/yulong/pvbg-thesis/HIT/mri_bones_release_v2/validation/female/NASI_3443_TS/mri_smpl.pkl"
    path_b = "/home/yulong/pvbg-thesis/HIT/hit_dataset_v1.0/repackaged/female_val.pkl"

    compare_smpl_datasets(path_a, path_b, subject_idx=0)
    #TEST_PATH = "mri_bones_release_v2/train/male/BH1470/per_part_pc/Femur_pc.ply"
    #cvisualize_sampling(TEST_PATH)
