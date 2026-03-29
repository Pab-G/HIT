"""Compute one-sided mean surface distance from GT bone point clouds to predicted meshes.

Supports two evaluation modes:
  - Whole-mesh: GT bone points -> entire predicted skeleton mesh
  - Per-bone: GT bone points -> corresponding predicted bone region only

For the specialist (our method), runs forward_rigged_bones to extract per-bone meshes
in v1/X-pose space, then transforms them to v2 posed space for comparison.

Usage:
    # Baselines only (fast, no GPU)
    PYTHONPATH=. python scripts/eval_bone_distance.py --skip_specialist

    # Full evaluation with specialist (needs GPU)
    PYTHONPATH=. python scripts/eval_bone_distance.py --specialist_exp OpusMale_4 --gender male
"""

import argparse
import json
import os
import pickle
import re

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

# ============================================================================
# Constants
# ============================================================================

DATA_ROOT = "mri_bones_release_v2/test"
V1_LOOKUP_PATH = "hit_dataset_v1.0/repackaged/v1_smpl_lookup.pkl"
MAPPING_PATH = "hit_dataset_v1.0/repackaged/mapping.json"

BONE_FILES = ["Femur_pc.ply", "Pelvis_pc.ply", "Humerus_pc.ply", "Radius-Ulna_pc.ply", "Tibia-Fibula_pc.ply"]
BONE_NAMES = ["Femur", "Pelvis", "Humerus", "Radius-Ulna", "Tibia-Fibula"]

# Pre-existing baseline meshes in each subject folder
BASELINE_METHODS = {
    "hit.ply": "HIT",
    "osso.ply": "OSSO",
    "skel.ply": "SKEL",
    "skel_j.ply": "SKEL-J",
}

# Vertex color -> bone name mapping for SKEL / SKEL-J per-bone evaluation
BONE_COLOR_MAP = {
    (31, 89, 208): "Femur",
    (85, 199, 240): "Pelvis",
    (187, 50, 94): "Humerus",
    (60, 50, 56): "Radius-Ulna",
    (216, 135, 64): "Tibia-Fibula",
}

# Methods that support per-bone evaluation via vertex colors
COLOR_PERBONE_METHODS = {"skel.ply": "SKEL", "skel_j.ply": "SKEL-J"}

N_SURFACE_SAMPLES = 100_000


# ============================================================================
# Utility functions
# ============================================================================

def procrustes_align(source_pts, target_pts):
    """Compute rigid alignment (R, t) such that target ≈ R @ source + t."""
    src_center = source_pts.mean(0)
    tgt_center = target_pts.mean(0)
    src_centered = source_pts - src_center
    tgt_centered = target_pts - tgt_center
    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = tgt_center - R @ src_center
    return R, t


def transform_mesh_v1_to_v2(mesh, R, t):
    """Transform mesh from v1 posed space to v2 posed space.

    Procrustes goes v2->v1: v1_pts = R @ v2_pts + t
    Inverse: v2_pts = R.T @ (v1_pts - t)
    """
    verts = np.array(mesh.vertices)
    v2_verts = (verts - t) @ R  # equivalent to R.T @ (verts - t).T then transpose
    return trimesh.Trimesh(vertices=v2_verts, faces=mesh.faces, process=False)


def extract_bone_submesh(mesh, bone_name, tolerance=2):
    """Extract submesh for a specific bone based on vertex colors."""
    target_rgb = None
    for color, name in BONE_COLOR_MAP.items():
        if name == bone_name:
            target_rgb = np.array(color, dtype=np.uint8)
            break
    if target_rgb is None:
        return None

    vertex_colors = np.array(mesh.visual.vertex_colors[:, :3])
    match = np.all(np.abs(vertex_colors.astype(int) - target_rgb.astype(int)) <= tolerance, axis=1)

    face_mask = match[mesh.faces].all(axis=1)
    if face_mask.sum() == 0:
        return None

    submesh = mesh.submesh([np.where(face_mask)[0]], append=True)
    return submesh


def compute_gt_to_mesh_distance(gt_points, mesh):
    """One-sided distances from GT points to mesh surface. Returns array of distances."""
    surface_pts, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_SAMPLES)
    tree = cKDTree(surface_pts)
    dists, _ = tree.query(gt_points, k=1)
    return dists


def evaluate_subject_whole(subj_path, mesh):
    """Evaluate whole-mesh distance for one subject. Returns per-bone mean distances (meters)."""
    gt_dir = os.path.join(subj_path, "per_part_pc")
    bone_dists = {}
    for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
        gt_path = os.path.join(gt_dir, bone_file)
        if not os.path.exists(gt_path):
            bone_dists[bone_name] = np.nan
            continue
        gt_pc = trimesh.load(gt_path)
        dists = compute_gt_to_mesh_distance(gt_pc.vertices, mesh)
        bone_dists[bone_name] = float(np.mean(dists))
    return bone_dists


def evaluate_subject_perbone_colors(subj_path, mesh):
    """Per-bone evaluation using vertex colors (SKEL/SKEL-J)."""
    gt_dir = os.path.join(subj_path, "per_part_pc")
    bone_dists = {}
    for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
        gt_path = os.path.join(gt_dir, bone_file)
        if not os.path.exists(gt_path):
            bone_dists[bone_name] = np.nan
            continue
        submesh = extract_bone_submesh(mesh, bone_name)
        if submesh is None or len(submesh.faces) == 0:
            bone_dists[bone_name] = np.nan
            continue
        gt_pc = trimesh.load(gt_path)
        dists = compute_gt_to_mesh_distance(gt_pc.vertices, submesh)
        bone_dists[bone_name] = float(np.mean(dists))
    return bone_dists


def evaluate_subject_perbone_meshes(subj_path, bone_meshes):
    """Per-bone evaluation using individual bone meshes (specialist)."""
    gt_dir = os.path.join(subj_path, "per_part_pc")
    bone_dists = {}
    for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
        gt_path = os.path.join(gt_dir, bone_file)
        if not os.path.exists(gt_path):
            bone_dists[bone_name] = np.nan
            continue
        if bone_name not in bone_meshes or bone_meshes[bone_name] is None:
            bone_dists[bone_name] = np.nan
            continue
        mesh = bone_meshes[bone_name]
        if len(mesh.faces) == 0:
            bone_dists[bone_name] = np.nan
            continue
        gt_pc = trimesh.load(gt_path)
        dists = compute_gt_to_mesh_distance(gt_pc.vertices, mesh)
        bone_dists[bone_name] = float(np.mean(dists))
    return bone_dists


def aggregate_bone_dists(bone_dists):
    """Convert per-bone distances dict to entry dict with mm values."""
    valid_dists = [d for d in bone_dists.values() if not np.isnan(d)]
    if not valid_dists:
        return None
    agg_dist_mm = np.mean(valid_dists) * 1000
    entry = {"agg_dist_mm": agg_dist_mm}
    for bn in BONE_NAMES:
        entry[f"{bn}_mm"] = bone_dists.get(bn, np.nan) * 1000
    return entry


# ============================================================================
# Specialist pipeline
# ============================================================================

def build_v2_to_v1_mapping():
    """Build mapping from v2 subject name -> v1 lookup key."""
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    v2_to_v1 = {}
    for v2_name, paths in mapping.items():
        m = re.search(r'/(male|female)/(train|test|val)/(\d+)\.gz', paths[1])
        if m:
            gender, v1_split, num_id = m.groups()
            v2_to_v1[v2_name] = f"{gender}_{v1_split}_{num_id}"
    return v2_to_v1


def generate_specialist_meshes(args, subject_paths):
    """Run forward_rigged_bones for each test subject, cache results.

    Returns dict: subject_name -> {bone_name: mesh_in_v2_space, 'merged': mesh}
    """
    import hit.hit_config as cg
    from hit.model.mysmpl import MySmpl
    from hit.utils.model import HitLoader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading standard HIT model...")
    hl = HitLoader.load_from_path(args.hit_path, args.hit_ckpt)
    hl.load()

    print(f"Loading specialist model ({args.specialist_exp})...")
    specialist_loader = HitLoader.from_expname(args.specialist_exp, ckpt_choice=args.ckpt_choice)
    specialist_loader.load()

    # Load v1 lookup and mapping
    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    # SMPL model for computing v2 meshes (for Procrustes)
    smpl_gender = args.gender if args.gender != "both" else "male"
    smpl_tool = MySmpl(model_path=cg.smplx_models_path, gender=smpl_gender).to(device)

    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")
        cache_dir = os.path.join(args.output_dir, subj)

        # Cache disabled — always recompute

        # Look up v1 data
        if subj not in v2_to_v1:
            print("  SKIP: no v2->v1 mapping")
            continue
        v1_key = v2_to_v1[subj]
        if v1_key not in v1_lookup:
            print(f"  SKIP: {v1_key} not in v1 lookup")
            continue
        v1_data = v1_lookup[v1_key]

        # Load v2 SMPL for Procrustes
        pkl_path = os.path.join(subj_path, "mri_smpl.pkl")
        with open(pkl_path, "rb") as f:
            v2_raw = pickle.load(f)

        v2_body_pose = v2_raw["pose"]
        if v2_body_pose.shape[0] == 63:
            v2_body_pose = np.concatenate([v2_body_pose, np.zeros(6, dtype=np.float32)])

        with torch.no_grad():
            v2_smpl_out = smpl_tool(
                betas=torch.tensor(v2_raw["betas"][:10]).unsqueeze(0).float().to(device),
                body_pose=torch.tensor(v2_body_pose).unsqueeze(0).float().to(device),
                global_orient=torch.tensor(v2_raw["global_rot"]).unsqueeze(0).float().to(device),
                transl=torch.tensor(v2_raw["trans"]).unsqueeze(0).float().to(device),
            )
        v2_verts = v2_smpl_out.vertices[0].cpu().numpy()
        v1_verts = v1_data["body_verts"].numpy()

        # Procrustes: v2 -> v1
        R, t = procrustes_align(v2_verts, v1_verts)

        # Run forward_rigged_bones with v1 params
        print("  Running forward_rigged_bones...")
        v1_betas = v1_data["betas"][:10].unsqueeze(0).float().to(device)
        v1_body_pose = v1_data["body_pose"].unsqueeze(0).float().to(device)
        v1_global_orient = v1_data["global_orient"].unsqueeze(0).float().to(device)
        v1_transl = v1_data["transl"].unsqueeze(0).float().to(device)

        with torch.no_grad():
            bone_meshes_v1 = hl.hit_model.forward_rigged_bones(
                specialist=specialist_loader.hit_model,
                betas=v1_betas,
                body_pose=v1_body_pose,
                global_orient=v1_global_orient,
                transl=v1_transl,
                mise_resolution0=64,
            )

        if not bone_meshes_v1:
            print("  WARNING: No bone meshes extracted")
            continue

        # Transform v1 -> v2 and cache
        os.makedirs(cache_dir, exist_ok=True)
        bone_meshes_v2 = {}
        mesh_list = []

        for bn in BONE_NAMES:
            if bn in bone_meshes_v1 and bn != "merged_skeleton":
                mesh_v2 = transform_mesh_v1_to_v2(bone_meshes_v1[bn], R, t)
                bone_meshes_v2[bn] = mesh_v2
                mesh_v2.export(os.path.join(cache_dir, f"{bn}.ply"))
                mesh_list.append(mesh_v2)
                print(f"  {bn}: {len(mesh_v2.vertices)} verts")

        if mesh_list:
            merged = trimesh.util.concatenate(mesh_list)
            merged.export(os.path.join(cache_dir, "merged.ply"))
            bone_meshes_v2["merged"] = merged
            print(f"  Merged: {len(merged.vertices)} verts")

        results[subj] = bone_meshes_v2

    return results


# ============================================================================
# Specialist classification accuracy
# ============================================================================

def evaluate_specialist(args, subject_paths):
    """Evaluate bone classification accuracy of the specialist on GT point clouds.

    For each test subject:
    1. Procrustes-align v2 SMPL -> v1 SMPL
    2. Transform GT bone point clouds from v2 posed space to v1 posed space
    3. Query the specialist to classify each GT point
    4. Report per-bone and overall accuracy
    """
    import hit.hit_config as cg
    from hit.model.deformer import skinning
    from hit.model.mysmpl import MySmpl
    from hit.utils.model import HitLoader
    from hit.utils.smpl_utils import get_skinning_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading standard HIT model...")
    hl = HitLoader.load_from_path(args.hit_path, args.hit_ckpt)
    hl.load()
    hit_model = hl.hit_model

    print(f"Loading specialist model ({args.specialist_exp})...")
    specialist_loader = HitLoader.from_expname(args.specialist_exp, ckpt_choice=args.ckpt_choice)
    specialist_loader.load()
    specialist = specialist_loader.hit_model

    specialist_labels = list(specialist.train_cfg.mri_labels)
    print(f"Specialist labels: {specialist_labels}")

    # Load v1 lookup and mapping
    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    smpl_gender = args.gender if args.gender != "both" else "male"
    smpl_tool = MySmpl(model_path=cg.smplx_models_path, gender=smpl_gender).to(device)

    all_results = []
    batch_size = 50000

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")

        if subj not in v2_to_v1:
            print("  SKIP: no v2->v1 mapping")
            continue
        v1_key = v2_to_v1[subj]
        if v1_key not in v1_lookup:
            print(f"  SKIP: {v1_key} not in v1 lookup")
            continue
        v1_data = v1_lookup[v1_key]

        # Load v2 SMPL, compute Procrustes (v2 -> v1)
        pkl_path = os.path.join(subj_path, "mri_smpl.pkl")
        with open(pkl_path, "rb") as f:
            v2_raw = pickle.load(f)

        v2_body_pose = v2_raw["pose"]
        if v2_body_pose.shape[0] == 63:
            v2_body_pose = np.concatenate([v2_body_pose, np.zeros(6, dtype=np.float32)])

        with torch.no_grad():
            v2_smpl_out = smpl_tool(
                betas=torch.tensor(v2_raw["betas"][:10]).unsqueeze(0).float().to(device),
                body_pose=torch.tensor(v2_body_pose).unsqueeze(0).float().to(device),
                global_orient=torch.tensor(v2_raw["global_rot"]).unsqueeze(0).float().to(device),
                transl=torch.tensor(v2_raw["trans"]).unsqueeze(0).float().to(device),
            )
        v2_verts = v2_smpl_out.vertices[0].cpu().numpy()
        v1_verts = v1_data["body_verts"].numpy()
        R, t = procrustes_align(v2_verts, v1_verts)

        # v1 posed SMPL output (for skinning weights & unposing)
        v1_betas = v1_data["betas"][:10].unsqueeze(0).float().to(device)
        v1_body_pose = v1_data["body_pose"].unsqueeze(0).float().to(device)
        v1_global_orient = v1_data["global_orient"].unsqueeze(0).float().to(device)
        v1_transl = v1_data["transl"].unsqueeze(0).float().to(device)

        with torch.no_grad():
            smpl_output_posed = specialist.smpl.forward(
                betas=v1_betas,
                body_pose=v1_body_pose,
                global_orient=v1_global_orient,
                transl=v1_transl,
            )
            smpl_output_xpose = specialist.smpl.forward(
                betas=v1_betas,
                body_pose=specialist.smpl.x_cano().to(device),
                global_orient=None,
                transl=None,
            )
        posed_verts = smpl_output_posed.vertices[0].cpu().numpy()
        xpose_verts = smpl_output_xpose.vertices[0].cpu().numpy()

        # Load GT bone point clouds, transform to v1 posed space
        gt_dir = os.path.join(subj_path, "per_part_pc")
        all_points = []
        all_labels = []

        for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
            gt_path = os.path.join(gt_dir, bone_file)
            if not os.path.exists(gt_path):
                continue
            if bone_name not in specialist_labels:
                continue
            label_idx = specialist_labels.index(bone_name)

            gt_pc = trimesh.load(gt_path)
            pts_v2 = np.array(gt_pc.vertices)
            # v2 posed -> v1 posed: v1_pts = R @ v2_pts + t
            pts_v1_posed = (R @ pts_v2.T).T + t

            all_points.append(pts_v1_posed)
            all_labels.append(np.full(len(pts_v1_posed), label_idx, dtype=np.int64))

        if not all_points:
            print("  SKIP: no GT bone point clouds found")
            continue

        all_points = np.concatenate(all_points, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0)
        n_points = len(all_points)
        print(f"  {n_points} GT points across {len(set(all_labels))} bones")

        # Two-step LBS to transform GT points: v1 posed -> rest -> X-pose
        # Step 1: inverse-skin with posed tfs (posed -> rest)
        # Step 2: forward-skin with X-pose tfs (rest -> X-pose)
        # Then query specialist in X-pose space (matching forward_rigged_bones).
        points_torch = torch.FloatTensor(all_points).unsqueeze(0).to(device)
        all_preds = []

        posed_tfs = smpl_output_posed.tfs   # [1, J, 4, 4]
        xpose_tfs = smpl_output_xpose.tfs   # [1, J, 4, 4]

        with torch.no_grad():
            # Step 1: posed -> rest (inverse LBS with posed bone transforms)
            sw_posed, _ = get_skinning_weights(all_points, posed_verts, specialist.smpl)
            sw_posed = torch.FloatTensor(sw_posed).to(device)  # [N, J]
            tfs_posed_exp = posed_tfs[0].unsqueeze(0).expand(n_points, -1, -1, -1)
            pts_rest = skinning(points_torch[0], sw_posed, tfs_posed_exp, inverse=True)

            # Step 2: rest -> X-pose (forward LBS with X-pose bone transforms)
            # Re-compute skinning weights relative to rest-pose SMPL for accuracy
            smpl_output_rest = specialist.smpl.forward(
                betas=v1_betas,
                body_pose=torch.zeros_like(v1_body_pose),
                global_orient=None,
                transl=None,
            )
            rest_verts = smpl_output_rest.vertices[0].cpu().numpy()
            sw_rest, _ = get_skinning_weights(pts_rest.cpu().numpy(), rest_verts, specialist.smpl)
            sw_rest = torch.FloatTensor(sw_rest).to(device)  # [N, J]
            tfs_xpose_exp = xpose_tfs[0].unsqueeze(0).expand(n_points, -1, -1, -1)
            pts_xpose = skinning(pts_rest, sw_rest, tfs_xpose_exp, inverse=False)
            pts_xpose = pts_xpose.unsqueeze(0)  # [1, N, 3]

        print(f"  Transformed {n_points} GT points: posed -> rest -> X-pose")

        # Query specialist in X-pose space (same as forward_rigged_bones Stage 2)
        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                pts_batch = pts_xpose[:, i:i + batch_size]
                pts_np = pts_batch[0].cpu().numpy()

                sw_xpose, part_id = get_skinning_weights(
                    pts_np, xpose_verts, specialist.smpl
                )
                sw_xpose = torch.FloatTensor(sw_xpose).unsqueeze(0).to(device)

                output = specialist.query(
                    pts_batch, smpl_output_xpose,
                    eval_mode=True, unposed=True,
                    part_id=part_id, skinning_weights=sw_xpose,
                )

                pred_occ = output['pred_occ']  # [1, N, num_classes]
                probs = torch.softmax(pred_occ, dim=-1)
                preds = torch.argmax(probs, dim=-1)[0].cpu().numpy()
                all_preds.append(preds)

                # Debug: print mean probabilities for first batch
                if i == 0:
                    mean_probs = probs[0].mean(dim=0).cpu().numpy()
                    print("  Mean softmax probs (first batch): "
                          + ", ".join(f"{bn}: {p:.3f}" for bn, p in zip(specialist_labels, mean_probs)))

        all_preds = np.concatenate(all_preds, axis=0)

        # Visualise X-pose points coloured by prediction vs GT
        # import matplotlib.pyplot as plt
        # from matplotlib.lines import Line2D

        # bone_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        # pts_np_xpose = pts_xpose[0].cpu().numpy()

        # fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})

        # for ax, labels_arr, title in zip(axes, [all_labels, all_preds], ['GT labels', 'Predicted']):
        #     for label_idx, bone_name in enumerate(specialist_labels):
        #         mask = labels_arr == label_idx
        #         if mask.sum() == 0:
        #             continue
        #         ax.scatter(
        #             pts_np_xpose[mask, 0], pts_np_xpose[mask, 1], pts_np_xpose[mask, 2],
        #             s=0.5, c=bone_colors[label_idx % len(bone_colors)], alpha=0.4,
        #             label=bone_name,
        #         )
        #     ax.set_title(f'{title} — {subj}')
        #     ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

        # legend_handles = [
        #     Line2D([0], [0], marker='o', color='w',
        #            markerfacecolor=bone_colors[i % len(bone_colors)], markersize=8,
        #            label=bn)
        #     for i, bn in enumerate(specialist_labels)
        # ]
        # fig.legend(handles=legend_handles, loc='lower center', ncol=len(specialist_labels))
        # plt.tight_layout(rect=[0, 0.06, 1, 1])
        # plt.savefig(f"bone_classification_{subj}.png", dpi=150)
        # print(f"  Saved bone_classification_{subj}.png")
        # plt.close(fig)

        # Accuracy
        correct = (all_preds == all_labels).sum()
        accuracy = correct / len(all_labels) * 100

        per_bone_acc = {}
        for label_idx, bone_name in enumerate(specialist_labels):
            mask = all_labels == label_idx
            if mask.sum() == 0:
                per_bone_acc[bone_name] = np.nan
                continue
            per_bone_acc[bone_name] = float((all_preds[mask] == label_idx).sum() / mask.sum() * 100)

        print(f"  Overall accuracy: {accuracy:.1f}%")
        for bn, acc in per_bone_acc.items():
            print(f"    {bn}: {acc:.1f}%")

        all_results.append({
            "subject": subj, "gender": gender,
            "n_points": n_points, "accuracy": float(accuracy),
            "per_bone": per_bone_acc,
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("  CLASSIFICATION ACCURACY SUMMARY")
    print(f"{'=' * 80}")
    header = f"{'Subject':<25} | {'N pts':>7} | {'Overall':>7} | "
    header += " | ".join(f"{bn[:8]:>8}" for bn in specialist_labels) + "  (%)"
    print(header)
    print("-" * 80)
    for r in all_results:
        bone_strs = " | ".join(
            f"{r['per_bone'].get(bn, np.nan):8.1f}" for bn in specialist_labels
        )
        print(f"{r['subject']:<25} | {r['n_points']:>7} | {r['accuracy']:>6.1f}% | {bone_strs}")

    if all_results:
        mean_acc = np.mean([r["accuracy"] for r in all_results])
        per_bone_means = {}
        for bn in specialist_labels:
            vals = [r["per_bone"][bn] for r in all_results if not np.isnan(r["per_bone"].get(bn, np.nan))]
            per_bone_means[bn] = np.mean(vals) if vals else np.nan
        bone_strs = " | ".join(f"{per_bone_means[bn]:8.1f}" for bn in specialist_labels)
        print("-" * 80)
        print(f"{'Mean':<25} | {'':>7} | {mean_acc:>6.1f}% | {bone_strs}")

    return all_results


# ============================================================================
# Full HIT + Specialist pipeline (matching forward_rigged_bones)
# ============================================================================

def full_hit_specialist_pipeline(args, subject_paths):
    """Evaluate bone classification using the full two-stage pipeline.

    For each test subject:
    1. Procrustes v2 -> v1, then two-step LBS to get GT points into v1 X-pose
    2. Query standard HIT on X-pose points to identify bone tissue (BONE = class 3)
    3. Feed only BT points to the specialist for bone-type classification
    4. Compare specialist predictions to GT labels, report accuracy
    """
    import hit.hit_config as cg
    from hit.model.deformer import skinning
    from hit.model.mysmpl import MySmpl
    from hit.utils.model import HitLoader
    from hit.utils.smpl_utils import get_skinning_weights

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    print("Loading standard HIT model...")
    hl = HitLoader.load_from_path(args.hit_path, args.hit_ckpt)
    hl.load()
    hit_model = hl.hit_model

    print(f"Loading specialist model ({args.specialist_exp})...")
    specialist_loader = HitLoader.from_expname(args.specialist_exp, ckpt_choice=args.ckpt_choice)
    specialist_loader.load()
    specialist = specialist_loader.hit_model

    specialist_labels = list(specialist.train_cfg.mri_labels)
    print(f"Specialist labels: {specialist_labels}")

    # Load v1 lookup and mapping
    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    smpl_gender = args.gender if args.gender != "both" else "male"
    smpl_tool = MySmpl(model_path=cg.smplx_models_path, gender=smpl_gender).to(device)

    all_results = []
    batch_size = 50000

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")

        if subj not in v2_to_v1:
            print("  SKIP: no v2->v1 mapping")
            continue
        v1_key = v2_to_v1[subj]
        if v1_key not in v1_lookup:
            print(f"  SKIP: {v1_key} not in v1 lookup")
            continue
        v1_data = v1_lookup[v1_key]

        # ---- Procrustes v2 -> v1 ----
        pkl_path = os.path.join(subj_path, "mri_smpl.pkl")
        with open(pkl_path, "rb") as f:
            v2_raw = pickle.load(f)

        v2_body_pose = v2_raw["pose"]
        if v2_body_pose.shape[0] == 63:
            v2_body_pose = np.concatenate([v2_body_pose, np.zeros(6, dtype=np.float32)])

        with torch.no_grad():
            v2_smpl_out = smpl_tool(
                betas=torch.tensor(v2_raw["betas"][:10]).unsqueeze(0).float().to(device),
                body_pose=torch.tensor(v2_body_pose).unsqueeze(0).float().to(device),
                global_orient=torch.tensor(v2_raw["global_rot"]).unsqueeze(0).float().to(device),
                transl=torch.tensor(v2_raw["trans"]).unsqueeze(0).float().to(device),
            )
        v2_verts = v2_smpl_out.vertices[0].cpu().numpy()
        v1_verts = v1_data["body_verts"].numpy()
        R, t = procrustes_align(v2_verts, v1_verts)

        # ---- SMPL outputs ----
        v1_betas = v1_data["betas"][:10].unsqueeze(0).float().to(device)
        v1_body_pose = v1_data["body_pose"].unsqueeze(0).float().to(device)
        v1_global_orient = v1_data["global_orient"].unsqueeze(0).float().to(device)
        v1_transl = v1_data["transl"].unsqueeze(0).float().to(device)

        with torch.no_grad():
            smpl_output_posed = hit_model.smpl.forward(
                betas=v1_betas,
                body_pose=v1_body_pose,
                global_orient=v1_global_orient,
                transl=v1_transl,
            )
            smpl_output_xpose = hit_model.smpl.forward(
                betas=v1_betas,
                body_pose=hit_model.smpl.x_cano().to(device),
                global_orient=None,
                transl=None,
            )
            smpl_output_rest = hit_model.smpl.forward(
                betas=v1_betas,
                body_pose=torch.zeros_like(v1_body_pose),
                global_orient=None,
                transl=None,
            )
        posed_verts = smpl_output_posed.vertices[0].cpu().numpy()
        xpose_verts = smpl_output_xpose.vertices[0].cpu().numpy()
        rest_verts = smpl_output_rest.vertices[0].cpu().numpy()

        # ---- Load GT bone point clouds, transform v2 posed -> v1 posed ----
        gt_dir = os.path.join(subj_path, "per_part_pc")
        all_points = []
        all_labels = []

        for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
            gt_path = os.path.join(gt_dir, bone_file)
            if not os.path.exists(gt_path):
                continue
            if bone_name not in specialist_labels:
                continue
            label_idx = specialist_labels.index(bone_name)

            gt_pc = trimesh.load(gt_path)
            pts_v2 = np.array(gt_pc.vertices)
            pts_v1_posed = (R @ pts_v2.T).T + t

            all_points.append(pts_v1_posed)
            all_labels.append(np.full(len(pts_v1_posed), label_idx, dtype=np.int64))

        if not all_points:
            print("  SKIP: no GT bone point clouds found")
            continue

        all_points = np.concatenate(all_points, axis=0).astype(np.float32)
        all_labels = np.concatenate(all_labels, axis=0)
        n_points = len(all_points)
        print(f"  {n_points} GT points across {len(set(all_labels))} bones")

        # ---- Two-step LBS: v1 posed -> rest -> X-pose ----
        points_torch = torch.FloatTensor(all_points).unsqueeze(0).to(device)
        posed_tfs = smpl_output_posed.tfs
        xpose_tfs = smpl_output_xpose.tfs

        with torch.no_grad():
            # Step 1: posed -> rest
            sw_posed, _ = get_skinning_weights(all_points, posed_verts, hit_model.smpl)
            sw_posed = torch.FloatTensor(sw_posed).to(device)
            tfs_posed_exp = posed_tfs[0].unsqueeze(0).expand(n_points, -1, -1, -1)
            pts_rest = skinning(points_torch[0], sw_posed, tfs_posed_exp, inverse=True)

            # Step 2: rest -> X-pose
            sw_rest, _ = get_skinning_weights(pts_rest.cpu().numpy(), rest_verts, hit_model.smpl)
            sw_rest = torch.FloatTensor(sw_rest).to(device)
            tfs_xpose_exp = xpose_tfs[0].unsqueeze(0).expand(n_points, -1, -1, -1)
            pts_xpose = skinning(pts_rest, sw_rest, tfs_xpose_exp, inverse=False)
            pts_xpose = pts_xpose.unsqueeze(0)  # [1, N, 3]

        print(f"  Transformed {n_points} GT points: posed -> rest -> X-pose")

        # ---- Stage 1: Query standard HIT to identify bone tissue ----
        print("  Stage 1: Querying HIT for bone tissue classification...")
        bt_mask = np.zeros(n_points, dtype=bool)

        with torch.no_grad():
            for i in range(0, n_points, batch_size):
                pts_batch = pts_xpose[:, i:i + batch_size]
                pts_np = pts_batch[0].cpu().numpy()
                batch_len = len(pts_np)

                sw, part_id = get_skinning_weights(pts_np, xpose_verts, hit_model.smpl)
                sw = torch.FloatTensor(sw).unsqueeze(0).to(device)

                output = hit_model.query(
                    pts_batch, smpl_output_xpose,
                    eval_mode=True, unposed=True,
                    part_id=part_id, skinning_weights=sw,
                )

                pred_occ = output['pred_occ']  # [1, N, 4] for [NO, LT, AT, BONE]
                probs = torch.softmax(pred_occ, dim=-1)
                predicted_class = torch.argmax(probs, dim=-1)[0].cpu().numpy()
                bt_mask[i:i + batch_len] = (predicted_class == 3)  # BONE = index 3

        n_bt = bt_mask.sum()
        print(f"  HIT classified {n_bt}/{n_points} points as bone tissue ({n_bt/n_points*100:.1f}%)")

        if n_bt == 0:
            print("  SKIP: no bone tissue points found by HIT")
            continue

        # ---- Stage 2: Query specialist only on BT points ----
        print("  Stage 2: Querying specialist on bone tissue points...")
        pts_bt = pts_xpose[:, bt_mask]  # [1, N_bt, 3]
        labels_bt = all_labels[bt_mask]
        n_bt_pts = int(n_bt)

        all_preds_bt = []
        with torch.no_grad():
            for i in range(0, n_bt_pts, batch_size):
                pts_batch = pts_bt[:, i:i + batch_size]
                pts_np = pts_batch[0].cpu().numpy()

                sw, part_id = get_skinning_weights(pts_np, xpose_verts, specialist.smpl)
                sw = torch.FloatTensor(sw).unsqueeze(0).to(device)

                output = specialist.query(
                    pts_batch, smpl_output_xpose,
                    eval_mode=True, unposed=True,
                    part_id=part_id, skinning_weights=sw,
                )

                pred_occ = output['pred_occ']  # [1, N, num_classes]
                probs = torch.softmax(pred_occ, dim=-1)
                preds = torch.argmax(probs, dim=-1)[0].cpu().numpy()
                all_preds_bt.append(preds)
                
                
                # ######
                # import matplotlib.pyplot as plt
                # from matplotlib.lines import Line2D
                # bone_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                # pts_bt_np = pts_bt[0].cpu().numpy()  # shape [N, 3]
                # fig, ax = plt.subplots(figsize=(7, 6))
                # for label_idx, bone_name in enumerate(specialist_labels):
                #     mask = preds == label_idx
                #     if mask.sum() == 0:
                #         continue
                #     ax.scatter(
                #         pts_bt_np[mask, 0], pts_bt_np[mask, 1],
                #         s=1, c=bone_colors[label_idx % len(bone_colors)], alpha=0.6, label=bone_name
                #     )
                # ax.set_title(f'Bone Class Predictions (Front View - {subj})')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # legend_handles = [
                #     Line2D([0], [0], marker='o', color='w',
                #            markerfacecolor=bone_colors[i % len(bone_colors)], markersize=8,
                #            label=bn)
                #     for i, bn in enumerate(specialist_labels)
                # ]
                # ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(specialist_labels))
                # plt.tight_layout()
                # plt.show()
                # #from IPython import embed; embed(); exit()
                
                # ##### Ground truth point cloud Transfomation here: 
                # import matplotlib.pyplot as plt
                # from matplotlib.lines import Line2D
                # bone_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
                # pts_gt_xpose = pts_xpose[0].cpu().numpy()  # shape [N, 3]
                # fig, ax = plt.subplots(figsize=(7, 6))
                # for label_idx, bone_name in enumerate(specialist_labels):
                #     mask = all_labels == label_idx
                #     if mask.sum() == 0:
                #         continue
                #     ax.scatter(
                #         pts_gt_xpose[mask, 0], pts_gt_xpose[mask, 1],
                #         s=1, c=bone_colors[label_idx % len(bone_colors)], alpha=0.6, label=bone_name
                #     )
                # ax.set_title('GT Bone Groups (X-pose)')
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # legend_handles = [
                #     Line2D([0], [0], marker='o', color='w',
                #         markerfacecolor=bone_colors[i % len(bone_colors)], markersize=8,
                #         label=bn)
                #     for i, bn in enumerate(specialist_labels)
                # ]
                # ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=len(specialist_labels))
                # plt.tight_layout()
                # plt.show()
                # exit()
                # ######
                if i == 0:
                    mean_probs = probs[0].mean(dim=0).cpu().numpy()
                    print("  Mean softmax probs (BT points, first batch): "
                          + ", ".join(f"{bn}: {p:.3f}" for bn, p in zip(specialist_labels, mean_probs)))

        all_preds_bt = np.concatenate(all_preds_bt, axis=0)

        # ---- Accuracy (on BT points only) ----
        correct = (all_preds_bt == labels_bt).sum()
        accuracy = correct / len(labels_bt) * 100

        per_bone_acc = {}
        for label_idx, bone_name in enumerate(specialist_labels):
            mask = labels_bt == label_idx
            if mask.sum() == 0:
                per_bone_acc[bone_name] = np.nan
                continue
            per_bone_acc[bone_name] = float((all_preds_bt[mask] == label_idx).sum() / mask.sum() * 100)

        # Also report how many GT points HIT missed (not classified as bone)
        n_missed = n_points - n_bt
        hit_recall = n_bt / n_points * 100

        print(f"  HIT bone recall: {hit_recall:.1f}% ({n_bt}/{n_points})")
        print(f"  Specialist accuracy (on BT points): {accuracy:.1f}%")
        for bn, acc in per_bone_acc.items():
            print(f"    {bn}: {acc:.1f}%")

        all_results.append({
            "subject": subj, "gender": gender,
            "n_points": n_points, "n_bt": n_bt,
            "hit_bone_recall": float(hit_recall),
            "accuracy": float(accuracy),
            "per_bone": per_bone_acc,
        })

    # ---- Summary ----
    print(f"\n{'=' * 90}")
    print("  FULL PIPELINE CLASSIFICATION (HIT BT filter -> Specialist)")
    print(f"{'=' * 90}")
    header = (f"{'Subject':<25} | {'HIT BT%':>7} | {'Acc':>6} | "
              + " | ".join(f"{bn[:8]:>8}" for bn in specialist_labels) + "  (%)")
    print(header)
    print("-" * 90)
    for r in all_results:
        bone_strs = " | ".join(
            f"{r['per_bone'].get(bn, np.nan):8.1f}" for bn in specialist_labels
        )
        print(f"{r['subject']:<25} | {r['hit_bone_recall']:>6.1f}% | {r['accuracy']:>5.1f}% | {bone_strs}")

    if all_results:
        mean_acc = np.mean([r["accuracy"] for r in all_results])
        mean_recall = np.mean([r["hit_bone_recall"] for r in all_results])
        per_bone_means = {}
        for bn in specialist_labels:
            vals = [r["per_bone"][bn] for r in all_results if not np.isnan(r["per_bone"].get(bn, np.nan))]
            per_bone_means[bn] = np.mean(vals) if vals else np.nan
        bone_strs = " | ".join(f"{per_bone_means[bn]:8.1f}" for bn in specialist_labels)
        print("-" * 90)
        print(f"{'Mean':<25} | {mean_recall:>6.1f}% | {mean_acc:>5.1f}% | {bone_strs}")

    return all_results


# ============================================================================
# Reporting
# ============================================================================

def print_summary_table(title, results, method_names):
    """Print formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"{'Method':<12} | {'Gender':<7} | {'Mean':>6} {'Std':>6} {'Min':>6} {'Max':>6}  (mm)")
    print("-" * 80)
    for method_name in method_names:
        if method_name not in results:
            continue
        for gender in ["female", "male"]:
            entries = results[method_name].get(gender, [])
            if not entries:
                continue
            dists = [e["agg_dist_mm"] for e in entries]
            print(f"{method_name:<12} | {gender:<7} | "
                  f"{np.mean(dists):6.2f} {np.std(dists):6.2f} "
                  f"{np.min(dists):6.2f} {np.max(dists):6.2f}")


def save_csv(csv_path, results, method_names):
    """Save detailed per-subject per-bone results to CSV."""
    with open(csv_path, "w") as f:
        header = "method,gender,subject,dist_mean_mm," + ",".join(f"dist_{bn}_mm" for bn in BONE_NAMES)
        f.write(header + "\n")
        for method_name in method_names:
            if method_name not in results:
                continue
            for gender in ["female", "male"]:
                for entry in results[method_name].get(gender, []):
                    row = [method_name, gender, entry["subject"], f"{entry['agg_dist_mm']:.4f}"]
                    for bn in BONE_NAMES:
                        row.append(f"{entry[f'{bn}_mm']:.4f}")
                    f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bone distance evaluation")
    parser.add_argument("--hit_path", type=str, default="pretrained/hit_male",
                        help="Path to standard HIT checkpoint directory")
    parser.add_argument("--hit_ckpt", type=str, default="male_hit.ckpt",
                        help="Checkpoint filename for standard HIT")
    parser.add_argument("--specialist_exp", type=str, default="OpusMale_4",
                        help="Specialist experiment name")
    parser.add_argument("--ckpt_choice", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--output_dir", type=str, default="output/bone_eval",
                        help="Directory to cache generated bone meshes")
    parser.add_argument("--skip_specialist", action="store_true",
                        help="Skip specialist evaluation (baselines only)")
    parser.add_argument("--gender", type=str, default="both", choices=["male", "female", "both"])
    parser.add_argument("--eval_classification", action="store_true",
                        help="Run specialist classification accuracy on GT point clouds and exit")
    parser.add_argument("--eval_full_pipeline", action="store_true",
                        help="Run full HIT+specialist pipeline classification and exit")
    args = parser.parse_args()

    # Collect subject paths
    genders = ["female", "male"] if args.gender == "both" else [args.gender]
    subject_paths = []
    for gender in genders:
        gender_dir = os.path.join(DATA_ROOT, gender)
        if not os.path.exists(gender_dir):
            continue
        for subj in sorted(os.listdir(gender_dir)):
            subj_path = os.path.join(gender_dir, subj)
            if os.path.isdir(subj_path) and os.path.exists(os.path.join(subj_path, "per_part_pc")):
                subject_paths.append((gender, subj, subj_path))

    print(f"Found {len(subject_paths)} test subjects\n")

    # ---- Classification accuracy (early exit) ----
    if args.eval_full_pipeline:
        full_hit_specialist_pipeline(args, subject_paths)
        return
    if args.eval_classification:
        evaluate_specialist(args, subject_paths)
        return

    # ---- Generate specialist meshes (if not skipped) ----
    specialist_meshes = {}  # subj -> {bone_name: mesh, 'merged': mesh}
    if not args.skip_specialist:
        specialist_meshes = generate_specialist_meshes(args, subject_paths)

    # ---- WHOLE-MESH EVALUATION ----
    whole_results = {}

    # Baselines
    for method_file, method_name in BASELINE_METHODS.items():
        print(f"Evaluating {method_name} whole-mesh...")
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            mesh_path = os.path.join(subj_path, method_file)
            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path, process=False)
            if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
                continue
            bone_dists = evaluate_subject_whole(subj_path, mesh)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        whole_results[method_name] = method_results

    # Specialist merged mesh
    if specialist_meshes:
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            if subj not in specialist_meshes or "merged" not in specialist_meshes[subj]:
                continue
            merged = specialist_meshes[subj]["merged"]
            bone_dists = evaluate_subject_whole(subj_path, merged)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        whole_results["Ours"] = method_results

    # ---- PER-BONE EVALUATION ----
    perbone_results = {}

    # SKEL/SKEL-J via vertex colors
    for method_file, method_name in COLOR_PERBONE_METHODS.items():
        print(f"Evaluating {method_name} per-bone...")
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            mesh_path = os.path.join(subj_path, method_file)
            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path, process=False)
            bone_dists = evaluate_subject_perbone_colors(subj_path, mesh)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        perbone_results[method_name] = method_results

    # Specialist individual bones
    if specialist_meshes:
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            if subj not in specialist_meshes:
                continue
            bone_meshes = {bn: specialist_meshes[subj].get(bn) for bn in BONE_NAMES}
            bone_dists = evaluate_subject_perbone_meshes(subj_path, bone_meshes)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        perbone_results["Ours"] = method_results

    # ---- REPORTING ----
    whole_method_names = list(BASELINE_METHODS.values()) + (["Ours"] if specialist_meshes else [])
    perbone_method_names = list(COLOR_PERBONE_METHODS.values()) + (["Ours"] if specialist_meshes else [])

    print_summary_table("WHOLE-MESH EVALUATION (GT -> entire skeleton)", whole_results, whole_method_names)
    print_summary_table("PER-BONE EVALUATION (GT bone -> predicted bone only)", perbone_results, perbone_method_names)

    save_csv("bone_distance_results_whole.csv", whole_results, whole_method_names)
    save_csv("bone_distance_results_perbone.csv", perbone_results, perbone_method_names)

    # ---- SANITY CHECK: Ours merged vs existing HIT ----
    if "Ours" in whole_results and "HIT" in whole_results:
        print(f"\n{'=' * 80}")
        print("  SANITY CHECK: Ours (merged) vs existing HIT")
        print(f"{'=' * 80}")
        print(f"{'Subject':<25} | {'HIT':>8} | {'Ours':>8} | {'Diff':>8}  (mm)")
        print("-" * 60)
        for gender in ["female", "male"]:
            hit_by_subj = {e["subject"]: e["agg_dist_mm"] for e in whole_results["HIT"].get(gender, [])}
            for entry in whole_results["Ours"].get(gender, []):
                subj = entry["subject"]
                ours = entry["agg_dist_mm"]
                hit = hit_by_subj.get(subj, np.nan)
                diff = ours - hit
                print(f"{subj:<25} | {hit:8.2f} | {ours:8.2f} | {diff:+8.2f}")


if __name__ == "__main__":
    main()
