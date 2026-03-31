import argparse
import json
import os
import pickle
import re

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

DATA_ROOT = "mri_bones_release_v2/test"
V1_LOOKUP_PATH = "hit_dataset_v1.0/repackaged/v1_smpl_lookup.pkl"
MAPPING_PATH = "hit_dataset_v1.0/repackaged/mapping.json"

BONE_FILES = ["Femur_pc.ply", "Pelvis_pc.ply", "Humerus_pc.ply", "Radius-Ulna_pc.ply", "Tibia-Fibula_pc.ply"]
BONE_NAMES = ["Femur", "Pelvis", "Humerus", "Radius-Ulna", "Tibia-Fibula"]

BASELINE_METHODS = {
    "hit.ply": "HIT",
    "osso.ply": "OSSO",
    "skel.ply": "SKEL",
    "skel_j.ply": "SKEL-J",
}

BONE_COLOR_MAP = {
    (31, 89, 208): "Femur",
    (85, 199, 240): "Pelvis",
    (187, 50, 94): "Humerus",
    (60, 50, 56): "Radius-Ulna",
    (216, 135, 64): "Tibia-Fibula",
}

COLOR_PERBONE_METHODS = {"skel.ply": "SKEL", "skel_j.ply": "SKEL-J"}

N_SURFACE_SAMPLES = 100_000
BATCH_SIZE = 50_000


# Utility functions ---------------------------------------------------------------


def procrustes_align(source_pts, target_pts):
    """Compute rigid alignment (R, t) such that target ~ R @ source + t."""
    src_center = source_pts.mean(0)
    tgt_center = target_pts.mean(0)
    H = (source_pts - src_center).T @ (target_pts - tgt_center)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, np.sign(d)]) @ U.T
    t = tgt_center - R @ src_center
    return R, t


def transform_mesh_v1_to_v2(mesh, R, t):
    """Inverse Procrustes: v2_pts = R.T @ (v1_pts - t)."""
    v2_verts = (np.array(mesh.vertices) - t) @ R
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
    return mesh.submesh([np.where(face_mask)[0]], append=True)


def compute_gt_to_mesh_distance(gt_points, mesh):
    """One-sided distances from GT points to mesh surface."""
    surface_pts, _ = trimesh.sample.sample_surface(mesh, N_SURFACE_SAMPLES)
    tree = cKDTree(surface_pts)
    dists, _ = tree.query(gt_points, k=1)
    return dists


# Shared setup helpers ------------------------------------------------------------


def build_v2_to_v1_mapping():
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    v2_to_v1 = {}
    for v2_name, paths in mapping.items():
        m = re.search(r'/(male|female)/(train|test|val)/(\d+)\.gz', paths[1])
        if m:
            gender, v1_split, num_id = m.groups()
            v2_to_v1[v2_name] = f"{gender}_{v1_split}_{num_id}"
    return v2_to_v1


def load_models(args):
    """Load HIT and specialist models. Returns (hit_model, specialist, smpl_tool, device)."""
    import hit.hit_config as cg
    from hit.model.mysmpl import MySmpl
    from hit.utils.model import HitLoader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading standard HIT model...")
    hl = HitLoader.load_from_path(args.hit_path, args.hit_ckpt)
    hl.load()

    print(f"Loading specialist model ({args.specialist_exp})...")
    sl = HitLoader.from_expname(args.specialist_exp, ckpt_choice=args.ckpt_choice)
    sl.load()

    smpl_gender = args.gender if args.gender != "both" else "male"
    smpl_tool = MySmpl(model_path=cg.smplx_models_path, gender=smpl_gender).to(device)

    return hl.hit_model, sl.hit_model, smpl_tool, device


def prepare_subject(subj, subj_path, v2_to_v1, v1_lookup, smpl_tool, device):
    """Resolve v1 data and compute Procrustes alignment.

    Returns (v1_data, R, t) or None if subject should be skipped.
    """
    if subj not in v2_to_v1:
        print("  SKIP: no v2->v1 mapping")
        return None
    v1_key = v2_to_v1[subj]
    if v1_key not in v1_lookup:
        print(f"  SKIP: {v1_key} not in v1 lookup")
        return None
    v1_data = v1_lookup[v1_key]

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

    return v1_data, R, t


def v1_params_to_device(v1_data, device):
    """Move v1 SMPL params to device with correct shape."""
    return {
        "betas": v1_data["betas"][:10].unsqueeze(0).float().to(device),
        "body_pose": v1_data["body_pose"].unsqueeze(0).float().to(device),
        "global_orient": v1_data["global_orient"].unsqueeze(0).float().to(device),
        "transl": v1_data["transl"].unsqueeze(0).float().to(device),
    }


def load_gt_bone_points(subj_path, bone_labels, R, t):
    """Load GT bone point clouds and transform from v2 posed to v1 posed space.

    Returns (points, labels) arrays or (None, None) if no data found.
    """
    gt_dir = os.path.join(subj_path, "per_part_pc")
    all_points, all_labels = [], []

    for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
        gt_path = os.path.join(gt_dir, bone_file)
        if not os.path.exists(gt_path) or bone_name not in bone_labels:
            continue
        label_idx = bone_labels.index(bone_name)
        gt_pc = trimesh.load(gt_path)
        pts_v1_posed = (R @ np.array(gt_pc.vertices).T).T + t
        all_points.append(pts_v1_posed)
        all_labels.append(np.full(len(pts_v1_posed), label_idx, dtype=np.int64))

    if not all_points:
        return None, None
    return np.concatenate(all_points).astype(np.float32), np.concatenate(all_labels)


def transform_posed_to_xpose(points, smpl_model, v1_params, device):
    """Two-step LBS: v1 posed -> rest -> X-pose.

    Returns (pts_xpose [1, N, 3], smpl_output_xpose).
    """
    from hit.model.deformer import skinning
    from hit.utils.smpl_utils import get_skinning_weights

    n = len(points)
    points_torch = torch.FloatTensor(points).to(device)

    with torch.no_grad():
        smpl_posed = smpl_model.forward(**v1_params)
        smpl_xpose = smpl_model.forward(
            betas=v1_params["betas"],
            body_pose=smpl_model.x_cano().to(device),
        )
        smpl_rest = smpl_model.forward(
            betas=v1_params["betas"],
            body_pose=torch.zeros_like(v1_params["body_pose"]),
        )

        # Step 1: posed -> rest (inverse LBS)
        sw_posed, _ = get_skinning_weights(points, smpl_posed.vertices[0].cpu().numpy(), smpl_model)
        sw_posed = torch.FloatTensor(sw_posed).to(device)
        tfs_posed = smpl_posed.tfs[0].unsqueeze(0).expand(n, -1, -1, -1)
        pts_rest = skinning(points_torch, sw_posed, tfs_posed, inverse=True)

        # Step 2: rest -> X-pose (forward LBS)
        sw_rest, _ = get_skinning_weights(pts_rest.cpu().numpy(), smpl_rest.vertices[0].cpu().numpy(), smpl_model)
        sw_rest = torch.FloatTensor(sw_rest).to(device)
        tfs_xpose = smpl_xpose.tfs[0].unsqueeze(0).expand(n, -1, -1, -1)
        pts_xpose = skinning(pts_rest, sw_rest, tfs_xpose, inverse=False)

    return pts_xpose.unsqueeze(0), smpl_xpose


def batch_classify(model, pts_xpose, smpl_output_xpose, xpose_verts):
    """Query model in batches, return predicted class indices."""
    from hit.utils.smpl_utils import get_skinning_weights

    n = pts_xpose.shape[1]
    device = pts_xpose.device
    all_preds = []

    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            pts_batch = pts_xpose[:, i:i + BATCH_SIZE]
            pts_np = pts_batch[0].cpu().numpy()

            sw, part_id = get_skinning_weights(pts_np, xpose_verts, model.smpl)
            sw = torch.FloatTensor(sw).unsqueeze(0).to(device)

            output = model.query(
                pts_batch, smpl_output_xpose,
                eval_mode=True, unposed=True,
                part_id=part_id, skinning_weights=sw,
            )

            probs = torch.softmax(output['pred_occ'], dim=-1)
            preds = torch.argmax(probs, dim=-1)[0].cpu().numpy()
            all_preds.append(preds)

            if i == 0:
                mean_probs = probs[0].mean(dim=0).cpu().numpy()
                labels = list(model.train_cfg.mri_labels)
                print("  Mean softmax probs (first batch): "
                      + ", ".join(f"{bn}: {p:.3f}" for bn, p in zip(labels, mean_probs)))

    return np.concatenate(all_preds)


def compute_classification_accuracy(preds, labels, label_names):
    """Returns (overall_accuracy_pct, {bone_name: accuracy_pct})."""
    accuracy = float((preds == labels).sum() / len(labels) * 100)
    per_bone = {}
    for idx, name in enumerate(label_names):
        mask = labels == idx
        if mask.sum() == 0:
            per_bone[name] = np.nan
        else:
            per_bone[name] = float((preds[mask] == idx).sum() / mask.sum() * 100)
    return accuracy, per_bone


# Distance evaluation -------------------------------------------------------------


def evaluate_bone_distances(subj_path, get_mesh_for_bone):
    """Evaluate GT-to-prediction distances per bone.

    get_mesh_for_bone(bone_name) should return a mesh or None.
    """
    gt_dir = os.path.join(subj_path, "per_part_pc")
    bone_dists = {}
    for bone_file, bone_name in zip(BONE_FILES, BONE_NAMES):
        gt_path = os.path.join(gt_dir, bone_file)
        if not os.path.exists(gt_path):
            bone_dists[bone_name] = np.nan
            continue
        mesh = get_mesh_for_bone(bone_name)
        if mesh is None or not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            bone_dists[bone_name] = np.nan
            continue
        gt_pc = trimesh.load(gt_path)
        dists = compute_gt_to_mesh_distance(gt_pc.vertices, mesh)
        bone_dists[bone_name] = float(np.mean(dists))
    return bone_dists


def aggregate_bone_dists(bone_dists):
    """Convert per-bone distances dict to entry dict with mm values."""
    valid = [d for d in bone_dists.values() if not np.isnan(d)]
    if not valid:
        return None
    entry = {"agg_dist_mm": np.mean(valid) * 1000}
    for bn in BONE_NAMES:
        entry[f"{bn}_mm"] = bone_dists.get(bn, np.nan) * 1000
    return entry


# Specialist mesh generation -------------------------------------------------------


def generate_specialist_meshes(args, subject_paths):
    """Run forward_rigged_bones for each test subject.

    Returns dict: subj -> {bone_name: mesh_in_v2_space, 'merged': mesh}.
    """
    hit_model, specialist, smpl_tool, device = load_models(args)

    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    results = {}
    os.makedirs(args.output_dir, exist_ok=True)

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")

        prep = prepare_subject(subj, subj_path, v2_to_v1, v1_lookup, smpl_tool, device)
        if prep is None:
            continue
        v1_data, R, t = prep
        v1p = v1_params_to_device(v1_data, device)

        print("  Running forward_rigged_bones...")
        with torch.no_grad():
            bone_meshes_v1 = hit_model.forward_rigged_bones(
                specialist=specialist,
                betas=v1p["betas"], body_pose=v1p["body_pose"],
                global_orient=v1p["global_orient"], transl=v1p["transl"],
                mise_resolution0=64,
            )

        if not bone_meshes_v1:
            print("  WARNING: No bone meshes extracted")
            continue

        cache_dir = os.path.join(args.output_dir, subj)
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


# Classification pipelines ---------------------------------------------------------


def evaluate_specialist(args, subject_paths):
    """Evaluate bone classification accuracy of the specialist on GT point clouds."""
    hit_model, specialist, smpl_tool, device = load_models(args)

    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    specialist_labels = list(specialist.train_cfg.mri_labels)
    print(f"Specialist labels: {specialist_labels}")

    all_results = []

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")

        prep = prepare_subject(subj, subj_path, v2_to_v1, v1_lookup, smpl_tool, device)
        if prep is None:
            continue
        v1_data, R, t = prep
        v1p = v1_params_to_device(v1_data, device)

        points, labels = load_gt_bone_points(subj_path, specialist_labels, R, t)
        if points is None:
            print("  SKIP: no GT bone point clouds found")
            continue
        print(f"  {len(points)} GT points across {len(set(labels))} bones")

        pts_xpose, smpl_xpose = transform_posed_to_xpose(points, specialist.smpl, v1p, device)
        xpose_verts = smpl_xpose.vertices[0].cpu().numpy()
        print(f"  Transformed {len(points)} GT points: posed -> rest -> X-pose")

        preds = batch_classify(specialist, pts_xpose, smpl_xpose, xpose_verts)
        accuracy, per_bone_acc = compute_classification_accuracy(preds, labels, specialist_labels)

        print(f"  Overall accuracy: {accuracy:.1f}%")
        for bn, acc in per_bone_acc.items():
            print(f"    {bn}: {acc:.1f}%")

        all_results.append({
            "subject": subj, "gender": gender,
            "n_points": len(points), "accuracy": accuracy,
            "per_bone": per_bone_acc,
        })

    print_classification_summary("CLASSIFICATION ACCURACY SUMMARY", all_results, specialist_labels)
    return all_results


def full_hit_specialist_pipeline(args, subject_paths):
    """Full HIT BT filter -> specialist classification pipeline."""
    hit_model, specialist, smpl_tool, device = load_models(args)

    with open(V1_LOOKUP_PATH, "rb") as f:
        v1_lookup = pickle.load(f)
    v2_to_v1 = build_v2_to_v1_mapping()

    specialist_labels = list(specialist.train_cfg.mri_labels)
    print(f"Specialist labels: {specialist_labels}")

    all_results = []

    for gender, subj, subj_path in subject_paths:
        print(f"\n--- {subj} ({gender}) ---")

        prep = prepare_subject(subj, subj_path, v2_to_v1, v1_lookup, smpl_tool, device)
        if prep is None:
            continue
        v1_data, R, t = prep
        v1p = v1_params_to_device(v1_data, device)

        points, labels = load_gt_bone_points(subj_path, specialist_labels, R, t)
        if points is None:
            print("  SKIP: no GT bone point clouds found")
            continue
        n_points = len(points)
        print(f"  {n_points} GT points across {len(set(labels))} bones")

        pts_xpose, smpl_xpose = transform_posed_to_xpose(points, hit_model.smpl, v1p, device)
        xpose_verts = smpl_xpose.vertices[0].cpu().numpy()
        print(f"  Transformed {n_points} GT points: posed -> rest -> X-pose")

        # Stage 1: HIT bone tissue filter
        print("  Stage 1: Querying HIT for bone tissue classification...")
        hit_preds = batch_classify(hit_model, pts_xpose, smpl_xpose, xpose_verts)
        bt_mask = (hit_preds == 3)  # BONE = index 3
        n_bt = int(bt_mask.sum())
        print(f"  HIT classified {n_bt}/{n_points} points as bone tissue ({n_bt / n_points * 100:.1f}%)")

        if n_bt == 0:
            print("  SKIP: no bone tissue points found by HIT")
            continue

        # Stage 2: Specialist on BT points only
        print("  Stage 2: Querying specialist on bone tissue points...")
        pts_bt = pts_xpose[:, bt_mask]
        labels_bt = labels[bt_mask]

        preds_bt = batch_classify(specialist, pts_bt, smpl_xpose, xpose_verts)
        accuracy, per_bone_acc = compute_classification_accuracy(preds_bt, labels_bt, specialist_labels)
        hit_recall = n_bt / n_points * 100

        print(f"  HIT bone recall: {hit_recall:.1f}% ({n_bt}/{n_points})")
        print(f"  Specialist accuracy (on BT points): {accuracy:.1f}%")
        for bn, acc in per_bone_acc.items():
            print(f"    {bn}: {acc:.1f}%")

        all_results.append({
            "subject": subj, "gender": gender,
            "n_points": n_points, "n_bt": n_bt,
            "hit_bone_recall": float(hit_recall),
            "accuracy": accuracy,
            "per_bone": per_bone_acc,
        })

    print_full_pipeline_summary(all_results, specialist_labels)
    return all_results


# Reporting ------------------------------------------------------------------------


def print_classification_summary(title, results, label_names):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    header = f"{'Subject':<25} | {'N pts':>7} | {'Overall':>7} | "
    header += " | ".join(f"{bn[:8]:>8}" for bn in label_names) + "  (%)"
    print(header)
    print("-" * 80)
    for r in results:
        bone_strs = " | ".join(f"{r['per_bone'].get(bn, np.nan):8.1f}" for bn in label_names)
        print(f"{r['subject']:<25} | {r['n_points']:>7} | {r['accuracy']:>6.1f}% | {bone_strs}")
    if results:
        mean_acc = np.mean([r["accuracy"] for r in results])
        per_bone_means = {bn: np.nanmean([r["per_bone"].get(bn, np.nan) for r in results]) for bn in label_names}
        bone_strs = " | ".join(f"{per_bone_means[bn]:8.1f}" for bn in label_names)
        print("-" * 80)
        print(f"{'Mean':<25} | {'':>7} | {mean_acc:>6.1f}% | {bone_strs}")


def print_full_pipeline_summary(results, label_names):
    print(f"\n{'=' * 90}")
    print("  FULL PIPELINE CLASSIFICATION (HIT BT filter -> Specialist)")
    print(f"{'=' * 90}")
    header = (f"{'Subject':<25} | {'HIT BT%':>7} | {'Acc':>6} | "
              + " | ".join(f"{bn[:8]:>8}" for bn in label_names) + "  (%)")
    print(header)
    print("-" * 90)
    for r in results:
        bone_strs = " | ".join(f"{r['per_bone'].get(bn, np.nan):8.1f}" for bn in label_names)
        print(f"{r['subject']:<25} | {r['hit_bone_recall']:>6.1f}% | {r['accuracy']:>5.1f}% | {bone_strs}")
    if results:
        mean_acc = np.mean([r["accuracy"] for r in results])
        mean_recall = np.mean([r["hit_bone_recall"] for r in results])
        per_bone_means = {bn: np.nanmean([r["per_bone"].get(bn, np.nan) for r in results]) for bn in label_names}
        bone_strs = " | ".join(f"{per_bone_means[bn]:8.1f}" for bn in label_names)
        print("-" * 90)
        print(f"{'Mean':<25} | {mean_recall:>6.1f}% | {mean_acc:>5.1f}% | {bone_strs}")


def print_summary_table(title, results, method_names):
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


# Main -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bone distance evaluation")
    parser.add_argument("--hit_path", type=str, default="pretrained/hit_male")
    parser.add_argument("--hit_ckpt", type=str, default="male_hit.ckpt")
    parser.add_argument("--specialist_exp", type=str, default="NewBranch")
    parser.add_argument("--ckpt_choice", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--output_dir", type=str, default="output/bone_eval")
    parser.add_argument("--skip_specialist", action="store_true")
    parser.add_argument("--gender", type=str, default="male", choices=["male", "female", "both"])
    parser.add_argument("--eval_classification", action="store_true")
    parser.add_argument("--eval_full_pipeline", action="store_true")
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

    # Classification modes (early exit)
    if args.eval_full_pipeline:
        full_hit_specialist_pipeline(args, subject_paths)
        return
    if args.eval_classification:
        evaluate_specialist(args, subject_paths)
        return

    # ---- Distance evaluation mode ----
    specialist_meshes = {}
    if not args.skip_specialist:
        specialist_meshes = generate_specialist_meshes(args, subject_paths)

    # Whole-mesh evaluation
    whole_results = {}

    for method_file, method_name in BASELINE_METHODS.items():
        print(f"Evaluating {method_name} whole-mesh...")
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            mesh_path = os.path.join(subj_path, method_file)
            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path, process=False)
            bone_dists = evaluate_bone_distances(subj_path, lambda _, m=mesh: m)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        whole_results[method_name] = method_results

    if specialist_meshes:
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            if subj not in specialist_meshes or "merged" not in specialist_meshes[subj]:
                continue
            merged = specialist_meshes[subj]["merged"]
            bone_dists = evaluate_bone_distances(subj_path, lambda _, m=merged: m)
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        whole_results["Ours"] = method_results

    # Per-bone evaluation
    perbone_results = {}

    for method_file, method_name in COLOR_PERBONE_METHODS.items():
        print(f"Evaluating {method_name} per-bone...")
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            mesh_path = os.path.join(subj_path, method_file)
            if not os.path.exists(mesh_path):
                continue
            mesh = trimesh.load(mesh_path, process=False)
            bone_dists = evaluate_bone_distances(subj_path, lambda bn, m=mesh: extract_bone_submesh(m, bn))
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        perbone_results[method_name] = method_results

    if specialist_meshes:
        method_results = {"female": [], "male": []}
        for gender, subj, subj_path in subject_paths:
            if subj not in specialist_meshes:
                continue
            bm = specialist_meshes[subj]
            bone_dists = evaluate_bone_distances(subj_path, lambda bn, bm=bm: bm.get(bn))
            entry = aggregate_bone_dists(bone_dists)
            if entry:
                entry["subject"] = subj
                method_results[gender].append(entry)
        perbone_results["Ours"] = method_results

    # Reporting
    whole_method_names = list(BASELINE_METHODS.values()) + (["Ours"] if specialist_meshes else [])
    perbone_method_names = list(COLOR_PERBONE_METHODS.values()) + (["Ours"] if specialist_meshes else [])

    print_summary_table("WHOLE-MESH EVALUATION (GT -> entire skeleton)", whole_results, whole_method_names)
    print_summary_table("PER-BONE EVALUATION (GT bone -> predicted bone only)", perbone_results, perbone_method_names)

    save_csv("output/bone_distance_results_whole.csv", whole_results, whole_method_names)
    save_csv("output/bone_distance_results_perbone.csv", perbone_results, perbone_method_names)

    # Sanity check
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
