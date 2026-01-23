import os
import sys

# 1. Remove the local directory from the path so it doesn't shadow 'logging'
# This stops Python from picking up hit/training/logging.py instead of the real library
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir in sys.path:
    sys.path.remove(current_dir)

# 2. Force the import of the real logging library
# Re-add the directory if you need it for other local imports later
sys.path.append(current_dir)

import collections
import glob
import os
import pickle as pkl
import time

import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import trimesh

import hit.hit_config as cg
from hit.model.mysmpl import MySmpl
from hit.training.mri_sampling_utils import vis_create_pc

# all the MRI will be padded with zero to this size
MRI_SHAPE = (256, 256, 128)  # (512,512,128) for mpi
BODY_ONLY = False  # When the whole MRI is sampled, only sample voxels in the body mask and around (dilatation is used)
SMPL_KEYS = ["betas", "body_pose", "global_orient", "transl"]
BONE_MAPPING = {
    "Femur_pc.ply": 3,
    "Pelvis_pc.ply": 4,
    "Humerus_pc.ply": 5,
    "Radius-Ulna_pc.ply": 6,
    "Tibia-Fibula_pc.ply": 7,
}
if BODY_ONLY:
    raise NotImplementedError("TODO: implement the composition computation ")


def visualize_dataloader_sample(dataset, index=0):
    import numpy as np
    import trimesh

    # This calls __getitem__, which splits 'mri_data_packed' into keys
    sample = dataset[index]

    # In __getitem__, points are in 'mri_points' and labels are in 'mri_occ'
    if "mri_points" not in sample:
        print("Error: 'mri_points' not found. Available keys:", sample.keys())
        return

    points = sample["mri_points"]
    labels = sample["mri_occ"]

    # Convert to numpy if they are tensors
    if torch.is_tensor(points):
        points = points.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy().astype(int)

    # Specialist Color Map:
    # 3:Femur(Red), 4:Pelvis(Green), 5:Humerus(Blue), 6:Radius(Yellow), 7:Tibia(Magenta)
    color_map = {
        1: [200, 200, 200, 255],  # Gray (Skin)
        3: [255, 0, 0, 255],  # Red
        4: [0, 255, 0, 255],  # Green
        5: [0, 0, 255, 255],  # Blue
        6: [255, 255, 0, 255],  # Yellow
        7: [255, 0, 255, 255],  # Magenta
    }

    colors = np.array([color_map.get(l, [0, 0, 0, 255]) for l in labels])

    print(
        f"--- Visualizing Specialist Sample: {sample.get('seq_names', 'Unknown')} ---"
    )
    print(f"Total Points: {len(points)}")
    for lid in np.unique(labels):
        print(f" - Class {lid}: {np.sum(labels == lid)} points")

    pc = trimesh.points.PointCloud(points, colors=colors)
    trimesh.Scene([pc]).show()


def pad_mri_z_vector(mri_vector):
    # pad a vecor to size MRI_SHAPE[2] and repeat the last slice value
    assert mri_vector.shape[0] < MRI_SHAPE[2], (
        "Input vector is larger than the target size MRI_SHAPE[2]. Make MRI_SHAPE[2] bigger to fix."
    )
    arr = mri_vector
    z_pad = MRI_SHAPE[2] - arr.shape[0]
    padded = F.pad(input=arr, pad=(0, 0, 0, z_pad), mode="constant", value=0)
    padded[arr.shape[1] :, :] = arr[-1, :]
    return padded


def sample_uniform_from_max(shape, max_val, padding):
    # Sample uniformly
    max_val = max_val + padding
    return torch.rand(shape).to(max_val.device) * max_val * 2 - max_val


def pad_mri(mri_data):
    size_err = f"Input vector of size{mri_data.shape} is larger than the target size MRI_SHAPE {MRI_SHAPE}. Make MRI_SHAPE bigger to fix."
    mri_data = mri_data[: MRI_SHAPE[0], : MRI_SHAPE[1], : MRI_SHAPE[2]]

    arr = mri_data
    x_pad = MRI_SHAPE[0] - arr.shape[0]
    y_pad = MRI_SHAPE[1] - arr.shape[1]
    z_pad = MRI_SHAPE[2] - arr.shape[2]

    padded = F.pad(
        input=arr, pad=(0, z_pad, 0, y_pad, 0, x_pad), mode="constant", value=0
    )
    return padded


def pad_gradient(gradient_data):
    """Pad a  CxWxDxHx3x3 array to the MRI_SHAPE"""
    size_err = f"Input vector of size{gradient_data.shape} is larger than the target size MRI_SHAPE {MRI_SHAPE}. Make MRI_SHAPE bigger to fix."

    # If the mri is just one slice too big, we remove the last slice. We don't want to double the MRI_SHAPE array size just for that.
    margin = 1
    assert gradient_data.shape[0] < MRI_SHAPE[0] + margin, size_err
    assert gradient_data.shape[1] < MRI_SHAPE[1] + margin, size_err
    assert gradient_data.shape[2] < MRI_SHAPE[2] + margin, size_err

    gradient_data = gradient_data[:, : MRI_SHAPE[0], : MRI_SHAPE[1], : MRI_SHAPE[2], :]

    arr = gradient_data
    x_pad = MRI_SHAPE[0] - arr.shape[0]
    y_pad = MRI_SHAPE[1] - arr.shape[1]
    z_pad = MRI_SHAPE[2] - arr.shape[2]

    padded = F.pad(
        input=arr,
        pad=(0, 0, 0, 0, 0, z_pad, 0, y_pad, 0, x_pad),
        mode="constant",
        value=0,
    )
    return padded


def list_preprocessed_files(data_root, genders: list):
    """List all preprocessed files in the data root. Return only the genders listed in genders."""

    assert os.path.isdir(data_root), f"Data root {data_root} does not exist."
    for g in genders:
        assert g in ["male", "female"]
    assert genders, "No gender listed to fetch the dataset"
    paths = []
    labels = []
    li = 0
    for dataset in ["cds"]:
        for gender in genders:
            folder = os.path.join(data_root, dataset, gender)
            os.path.isdir(folder), f"Folder {folder} does not exist."
            file_list = os.listdir(folder)
            path_list = [os.path.join(folder, filename) for filename in file_list]
            paths.extend(path_list)
            labels.extend([li] * len(path_list))
            li += 1

    paths.sort()
    return paths


def get_split_files(data_root, gender, split):
    paths = glob.glob(os.path.join(data_root, gender, split, "*.gz"))
    return paths


def _get_split_files(data_root, gender, split):
    data_version = cg.data_version
    print(f"\n Loading splits for data version {data_version} for gender {gender} \n")
    if data_version == "v4":
        split_file = os.path.join("./splits", f"split_mri_{gender}_{split}.txt")
    else:
        split_file = os.path.join(
            f"./splits{data_version}", f"split_mri_{gender}_{split}.txt"
        )

    assert os.path.exists(split_file), (
        f"Split file {split_file} does not exist, you can create it with generate_splits.py"
    )

    paths = []
    with open(split_file, "r") as f:
        for line in f:
            # print(data_root, line.strip())
            paths.append(os.path.join(data_root, line.strip()))

    return paths


def print_splits_files(data_root):
    for gender in ["female", "male"]:
        for split in ["train", "test", "val"]:
            paths = get_split_files(data_root, gender, split)
            print(f"Number of {split} samples for {gender}: {len(paths)}")
            for path in paths:
                print("\t", path)


# @varora
# function to normalize the mri values by min max
def process_mri_values(mri_values: np.ndarray, normalize=False):
    if normalize:
        # min-max normalization
        mri_values = (mri_values - np.min(mri_values)) / (
            np.max(mri_values) - np.min(mri_values)
        )
    mri_values = mri_values.astype(np.float32)
    return mri_values


class MRIDataset(torch.utils.data.Dataset):
    @torch.no_grad()
    def __init__(self, smpl_cfg, data_cfg, train_cfg, smpl_data, split):
        super().__init__()
        self.smpl_cfg = smpl_cfg
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        self.smpl_data = smpl_data
        self.split = split  # changes the batch nb of vertices

        self.data_keys = list(smpl_data.keys())
        self.smpl = MySmpl(model_path=cg.smplx_models_path, gender=smpl_cfg["gender"])

        self.faces = self.smpl.faces.copy()

        self.synthetic = data_cfg.synthetic

        self.bbox_padding = 1.125

        if not self.synthetic:
            # find index of a subject with fat for overfitting if needed
            self.lowest_b1_subject_idx = self.find_lowest_b1()
            name = self.smpl_data["seq_names"][self.lowest_b1_subject_idx]
            print(f"Lowest b1 subject idx: {self.lowest_b1_subject_idx}, name : {name}")

            self.highest_b1_subject_idx = self.find_highest_b1()
            name = self.smpl_data["seq_names"][self.highest_b1_subject_idx]
            print(
                f"Highest b1 subject idx: {self.highest_b1_subject_idx}, name : {name}"
            )

            self.__len__()

            self.can_points_dictionary = (
                self.sample_can_space()
            )  # We do this once for all the dataset
            self.can_hands_dictionary = (
                self.sample_can_hands()
            )  # We do this once for all the dataset

    @classmethod
    @torch.no_grad()
    def from_config(
        cls,
        smpl_cfg,
        data_cfg,
        train_cfg,
        split="train",
        filter_indices=None,
        force_recache=False,
    ):
        # 1. Internal Imports to prevent NameErrors and environment issues
        import glob

        import trimesh
        from tqdm import tqdm

        from hit.utils.smpl_utils import get_skinning_weights

        gender = smpl_cfg["gender"]

        # 2. Path Discovery
        base_dir = "/home/yulong/pvbg-thesis/HIT/mri_bones_release_v2"
        paths = glob.glob(os.path.join(base_dir, split, gender, "*"))
        print(
            f"--- Found {len(paths)} subjects in {os.path.join(base_dir, split, gender)} ---"
        )

        if len(paths) == 0:
            raise ValueError(
                f"Stop! No data found at {base_dir}. Ensure paths are correct."
            )

        # Metadata and Model Setup
        smpl_tool = MySmpl(model_path=cg.smplx_models_path, gender=gender)
        varying_size_keys = [
            "mri_points",
            "mri_occ",
            "skinning_weights",
            "mri_data_packed",
        ]
        dataset_list = collections.defaultdict(list)

        # 3. Processing Loop
        print("--- Loading Specialist Dataset with Blue Jitter Sampling ---")
        for pi, path in tqdm(enumerate(paths)):
            pkl_path = os.path.join(path, "mri_smpl.pkl")
            if not os.path.exists(pkl_path):
                continue

            with open(pkl_path, "rb") as f:
                raw_data = pkl.load(f)

            # --- POSE DIMENSION FIX (63 -> 69) ---
            body_pose_raw = raw_data["pose"]
            if body_pose_raw.shape[0] == 63:
                body_pose_69 = np.concatenate(
                    [body_pose_raw, np.zeros(6, dtype=np.float32)]
                )
            else:
                body_pose_69 = body_pose_raw

            # --- SAMPLING POINTS FROM PLY ---
            mri_points = []
            gt_occ = []

            # Bone Specialist Sampling (Labels 3-7)
            bone_folder = os.path.join(path, "per_part_pc")
            for bone_file, class_id in BONE_MAPPING.items():
                bone_path = os.path.join(bone_folder, bone_file)
                if os.path.exists(bone_path):
                    bone_data = trimesh.load(bone_path)
                    pts = bone_data.vertices

                    # --- ROBUST SAMPLING FIX ---
                    # Handles cases where bone points < population size
                    pop_size = len(pts)
                    requested_size = 3000
                    use_replace = pop_size < requested_size

                    idx = np.random.choice(
                        pop_size, requested_size, replace=use_replace
                    )
                    surface_samples = pts[idx]

                    # --- BLUE JITTER LOGIC ---
                    # Adds 8mm standard deviation noise to simulate internal volume
                    jitter = np.random.normal(0, 0.008, (requested_size, 3))
                    volumetric_pts = surface_samples + jitter

                    mri_points.append(volumetric_pts)
                    gt_occ.append(np.full(requested_size, class_id))

            if len(mri_points) == 0:
                continue

            # --- PREPARING PACKED DATA ---
            mri_pts_all = np.concatenate(mri_points, axis=0).astype(np.float32)
            occ_labels_all = np.concatenate(gt_occ, axis=0).astype(np.float32)

            #
            # Create SMPL mesh to calculate internal skinning weights
            smpl_out = smpl_tool(
                betas=torch.tensor(raw_data["betas"][:10]).unsqueeze(0),
                body_pose=torch.tensor(body_pose_69).unsqueeze(0),
                global_orient=torch.tensor(raw_data["global_rot"]).unsqueeze(0),
                transl=torch.tensor(raw_data["trans"]).unsqueeze(0),
            )

            # Calculate weights (Maps every 3D point to its nearest SMPL skeleton joints)
            weights, _ = get_skinning_weights(
                mri_pts_all, smpl_out.vertices[0].cpu().numpy(), smpl_tool
            )

            # Pack into the 33-column tensor HIT expects
            packed = np.zeros((mri_pts_all.shape[0], 33), dtype=np.float32)
            packed[:, 0:3] = mri_pts_all
            packed[:, 6] = occ_labels_all
            packed[:, 8] = (
                1.0  # body_mask (tells model these points are inside the body)
            )
            packed[:, 9:33] = weights

            # --- STORE IN BUCKET ---
            dataset_list["mri_data_packed"].append(torch.from_numpy(packed))
            dataset_list["mri_data_shape0"].append(mri_pts_all.shape[0])
            dataset_list["mri_data_shape1"].append(packed.shape[1])
            dataset_list["betas"].append(torch.tensor(raw_data["betas"][:10]))
            dataset_list["body_pose"].append(torch.tensor(body_pose_69))
            dataset_list["global_orient"].append(torch.tensor(raw_data["global_rot"]))
            dataset_list["transl"].append(torch.tensor(raw_data["trans"]))
            dataset_list["seq_names"].append(os.path.basename(path))

        # 4. Final Stacking
        data_stacked = {}
        for key, val in dataset_list.items():
            if isinstance(val[0], torch.Tensor) and key not in varying_size_keys:
                data_stacked[key] = torch.stack(val, dim=0)
            else:
                data_stacked[key] = val

        print(f"--- SUCCESS: Loaded {len(dataset_list['seq_names'])} subjects ---")
        return cls(smpl_cfg, data_cfg, train_cfg, data_stacked, split)

    @torch.no_grad()
    def __getitem__(self, idx, return_smpl=False, get_whole_mri=False):
        t1 = time.perf_counter()
        if self.synthetic:
            return self._getitem_synthetic()

        # t1 = time.perf_counter()
        subj_data = {key: self.smpl_data[key][idx] for key in self.smpl_data.keys()}

        for key in ["betas", "body_pose", "global_orient", "transl"]:
            if key not in subj_data:
                raise ValueError(f"{key} not in subj_data")
            assert key in subj_data, f"{key} not in torch_param"

        if self.train_cfg["comp0_out"]:
            mri_points = subj_data["mri_data_packed"][:, 0:3]
            body_mask = subj_data["mri_data_packed"][:, 8]
            mri_out_pts = mri_points[body_mask == False]
            idx = np.random.randint(0, mri_out_pts.shape[0], 6000)
            mri_out_pts_sampled = mri_out_pts[idx]
            subj_data.update(mri_out_pts=mri_out_pts_sampled)

        # if self.data_cfg.subjects == 'lowest_b1':
        #     idx = self.lowest_b1_subject_idx
        #     name = self.smpl_data['seq_names'][idx]
        # print(f'Lowest b1 subject idx: {self.lowest_b1_subject_idx}, name : {name}')

        # We need to compute smpl output for this subject
        # t1 = time.perf_counter()
        # smpl_data_batched = {key: subj_data[key][None] for key in SMPL_KEYS}

        # t2 = time.perf_counter()
        # print(f'Get item A {idx} took {t2-t1:.2f}s')

        # smpl_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in smpl_data.items()}  # remove B dim

        # import ipdb; ipdb.set_trace()
        # if return_smpl:
        #     subj_data.update({'smpl_output': smpl_output})
        if (
            self.train_cfg["to_train"] != "compression"
        ):  # We do not need to sample pts for the compression
            # import ipdb; ipdb.set_trace()
            if self.split == "test" or get_whole_mri is True:
                subj_data.update(self.sample_whole_mri(subj_data))

            else:
                if self.data_cfg["sampling_strategy"] == "mri":
                    # @varora
                    # updated to sample whole mri
                    subj_data = self.sample_whole_mri(
                        subj_data, nb_points=self.data_cfg["n_pts_mri"]
                    )

                else:
                    raise DeprecationWarning(
                        "Needs to precompute SMP L, which takes time so avoid it"
                    )

                    if self.data_cfg["sampling_strategy"] == "per_part":
                        subj_data.update(self.sample_points(smpl_output, subj_data))
                    elif self.data_cfg["sampling_strategy"] == "per_tissue":
                        subj_data.update(
                            self.sample_given_number(subj_data, smpl_output)
                        )
                    elif self.data_cfg["sampling_strategy"] == "boundary":
                        subj_data.update(self.sample_boundary(smpl_output, subj_data))
                    elif self.data_cfg["sampling_strategy"] == "local":
                        subj_data.update(
                            self.sample_tissue_per_part(smpl_output, subj_data)
                        )
                    else:
                        raise NotImplementedError(
                            f"Sampling strategy {self.data_cfg['sampling_strategy']} not implemented"
                        )

                if self.data_cfg["sample_can_points"]:
                    # subsample can_points_dictionary
                    idx_rand = np.random.randint(
                        0, self.can_points_dictionary["can_points"].shape[0], 6000
                    )
                    can_points_dict = {
                        key: val[idx_rand]
                        for key, val in self.can_points_dictionary.items()
                    }
                    subj_data.update(can_points_dict)

                if self.data_cfg["sample_can_hands"]:
                    subj_data.update(self.can_hands_dictionary)
        else:
            subj_data = {
                key: val
                for key, val in subj_data.items()
                if key
                not in [
                    "mri_data_packed",
                    "mri_data_shape1",
                    "mri_data_keys",
                    "mri_data_shape0",
                ]
            }

            # t2 = time.perf_counter()
        # print(f'Get item {idx} took {t2-t1:.2f}s')
        # print(subj_data.keys())
        return subj_data

    def sample_whole_mri(self, subj_data, nb_points=None):
        stacked_data = subj_data["mri_data_packed"]
        mri_points_nb = subj_data["mri_data_shape0"]
        mri_data_shape1 = subj_data["mri_data_shape1"]
        # mri_data_keys = subj_data['mri_data_keys']

        # import ipdb; ipdb.set_trace()
        # to_ret = dict(mri_points=subj_data['mri_points'],
        #               mri_occ=subj_data['mri_occ'],
        #               mri_coords=subj_data['mri_coords'],
        #               body_mask = subj_data['body_mask'],
        #               part_id = subj_data['part_id'],
        #               skinning_weights = subj_data['skinning_weights'])
        # to_ret = dict(mri_points=subj_data['mri_points'],
        #               mri_occ=subj_data['mri_occ'][..., None],
        #               mri_coords=subj_data['mri_coords'],
        #               body_mask = subj_data['body_mask'][..., None],
        #               part_id = subj_data['part_id'][..., None],
        #               skinning_weights = subj_data['skinning_weights'])

        # [print(val.shape) for key, val in to_ret.items()]
        # stacked_data = np.concatenate([val for key, val in to_ret.items()], axis=1)

        # This could be optimised by stacking the upper arrays
        t1 = time.perf_counter()
        if nb_points is not None:
            # sample random points inside the body
            # idx_bodymask = np.where(subj_data['body_mask']==1)[0] # 0.16 s
            # # import ipdb; ipdb.set_trace()
            idx_to_keep = np.random.randint(0, mri_points_nb, nb_points)  # 0.05 (0.22)
            # # idx_to_keep = np.random.choice(idx_bodymask.shape[0], nb_points, replace=False) # 3.35s
            # to_ret = {key: val[idx_bodymask][idx_to_keep] for key, val in to_ret.items()}# 3sec
            stacked_data = stacked_data[idx_to_keep]  # 3sec
            # print(f'Number of MRI points to evaluate: {to_ret["points"].shape[0]}')
        t2 = time.perf_counter()
        # print(f'Time to sample {nb_points} points: {t2-t1:.5f}s')

        # import ipdb; ipdb.set_trace()
        # todo retrieve the last row to be mnri value and put it in key `mri_values`
        # @varora
        # retrieve the last row to be mri values and put it in key `mri_values`

        # import ipdb; ipdb.set_trace()
        if subj_data["mri_data_packed"].shape[1] > 33:
            if self.train_cfg.mri_values is True:
                to_ret = dict(
                    mri_points=stacked_data[:, :3],
                    mri_coords=stacked_data[:, 3:6],
                    mri_occ=stacked_data[:, 6],
                    part_id=stacked_data[:, 7],
                    body_mask=stacked_data[:, 8],
                    skinning_weights=stacked_data[:, 9:33],
                    mri_values=stacked_data[:, 33:],
                )
            else:
                to_ret = dict(
                    mri_points=stacked_data[:, :3],
                    mri_coords=stacked_data[:, 3:6],
                    mri_occ=stacked_data[:, 6],
                    part_id=stacked_data[:, 7],
                    body_mask=stacked_data[:, 8],
                    skinning_weights=stacked_data[:, 9:33],
                )
        else:
            to_ret = dict(
                mri_points=stacked_data[:, :3],
                mri_coords=stacked_data[:, 3:6],
                mri_occ=stacked_data[:, 6],
                part_id=stacked_data[:, 7],
                body_mask=stacked_data[:, 8],
                skinning_weights=stacked_data[:, 9:],
            )

        assert to_ret["skinning_weights"].shape[1] == 24, (
            f"Wrong number of skinning weights: {to_ret['skinning_weights'].shape[1]}"
        )

        not_ret_keys = [
            "mri_data_packed",
            "mri_data_shape1",
            "mri_data_keys",
            "mri_data_shape0",
        ]
        to_ret.update(
            {key: val for key, val in subj_data.items() if key not in not_ret_keys}
        )

        return to_ret

    def sample_can_space(self):
        nb_outside_pts = self.data_cfg["nb_points_canspace"]
        nb_skin_pts = self.data_cfg["n_skin_pts"]

        smpl_can = self.smpl.forward_canonical()
        can_vertices = smpl_can.vertices.detach().cpu().numpy()
        can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

        # Uniform points
        max_val = can_vertices.max() + self.data_cfg["uniform_sampling_padding"]
        uniform_pts = (
            torch.rand((nb_outside_pts, 3)) * max_val * 2 - max_val
        )  # Sample uniformly
        # uniform_pts = center_on_voxel(uniform_pts.cpu().numpy(), smpl_data) # Align to voxel grid

        # import ipdb; ipdb.set_trace()

        # Skin surface points
        if nb_skin_pts > 0:
            points, faces = can_mesh.sample(nb_skin_pts, return_index=True)
            normals = can_mesh.face_normals[faces]

            surface_offset_min = self.data_cfg["surface_offset_min"]
            surface_offset_max = self.data_cfg["surface_offset_max"]
            offset = surface_offset_min + (
                surface_offset_max - surface_offset_min
            ) * np.abs(np.random.randn(nb_skin_pts, 1))
            surface_points = points + offset * normals
            surface_points = surface_points.astype(np.float32)
            # surface_points = center_on_voxel(surface_points.cpu().numpy(), smpl_data)

            can_pts = np.concatenate((uniform_pts, surface_points), axis=0)
        else:
            can_pts = uniform_pts

        # from leap.tools.libmesh import check_mesh_contains
        # can_occ = check_mesh_contains(can_mesh, can_pts).astype(np.float32)
        # can_occ = can_mesh.contains(can_pts)
        # gt_occ, body_mask = load_occupancy(smpl_data, can_pts, interp_order=0)

        # import ipdb; ipdb.set_trace()
        from pysdf import SDF

        f = SDF(
            can_mesh.vertices, can_mesh.faces
        )  # (num_vertices, 3) and (num_faces, 3)
        can_occ = f.contains(can_pts)

        return dict(can_points=can_pts, can_occ=can_occ)

    def sample_can_hands(self):
        nb_pts = self.data_cfg["n_points_hands"]

        smpl_can = self.smpl(betas=torch.zeros(1, 10).to(self.smpl.smpl.betas.device))
        can_vertices = smpl_can.vertices.detach().cpu().numpy()
        can_mesh = trimesh.Trimesh(can_vertices.squeeze(), self.faces, process=False)

        bone_trans = self.smpl.compute_bone_trans(
            smpl_can.full_pose, smpl_can.joints
        )  # 1,24,4,4
        bbox_min, bbox_max = self.smpl.get_bbox_bounds_trans(
            smpl_can.vertices, bone_trans
        )  # (B, K, 1, 3) [can space]
        n_parts = bbox_max.shape[1]

        #### Sample points inside local boxes

        bbox_size = (bbox_max - bbox_min).abs() * self.bbox_padding - 1e-3  # (B,K,1,3)
        bbox_center = (bbox_min + bbox_max) * 0.5
        bb_min = bbox_center - bbox_size * 0.5  # to account for padding

        ##### Sample points uniformly in the body bounding boxes
        uniform_points = (
            bb_min + torch.rand((1, n_parts, nb_pts, 3)) * bbox_size
        )  # [0,bs] (B,K,N,3)
        abs_transforms = torch.inverse(bone_trans)  # B,K,4,4
        uniform_points = (
            abs_transforms.reshape(1, n_parts, 1, 4, 4).repeat(1, 1, nb_pts, 1, 1)
            @ F.pad(uniform_points, [0, 1], "constant", 1.0).unsqueeze(-1)
        )[..., :3, 0]

        if self.data_cfg.sample_can_toes is False:
            hand_pts = uniform_points[0, 20:24]
        else:
            toes_pts = uniform_points[0, 10:12]
            hand_pts = torch.cat((toes_pts), dim=1)
        can_pts = hand_pts.reshape(-1, 3).numpy()
        # can_occ = can_mesh.contains(can_pts)

        # from leap.tools.libmesh import check_mesh_contains
        # can_occ = check_mesh_contains(can_mesh, can_pts).astype(np.float32)

        # gt_occ, body_mask = load_occupancy(smpl_data, can_pts.cpu().numpy(), interp_order=0)

        # import ipdb; ipdb.set_trace()
        from pysdf import SDF

        f = SDF(
            can_mesh.vertices, can_mesh.faces
        )  # (num_vertices, 3) and (num_faces, 3)
        can_occ = f.contains(can_pts)

        return dict(hands_can_points=can_pts, hands_can_occ=can_occ)

    def display_sample(self, idx, color_style="by_class", can_only=False):
        assert color_style in ["by_class", "by_skinning"]

        nb_beta = 10

        data = self.__getitem__(idx, return_smpl=True)
        points = data["mri_points"]
        occ = data["mri_occ"]

        if self.data_cfg["sample_can_points"]:
            can_points = data["can_points"]
            can_occ = data["can_occ"]

        if self.data_cfg["sample_can_hands"]:
            hands_can_points = data["hands_can_points"]
            hands_can_occ = data["hands_can_occ"]

        if color_style == "by_skinning":
            skinning_weights = data["skinning_weights"][:]
            from utils.smpl_utils import weights2colors

            skinning_color = weights2colors(skinning_weights)

        print(f"Number of points sampled: {points.shape[0]}")

        print(f"Visualizing sample {data['seq_names']}")
        if points.shape[0] > 50000:
            points = points[:-1:500]
            occ = occ[:-1:500]  # Don't display all the points otherwise it will crash

        # mri_points, mri_coords = sample_mri_pts(data, body_only=BODY_ONLY)
        # mri_points = mri_points[0:-1:100]
        # print(f'Number of MRI points displayed: {mri_points.shape[0]}')

        # smpl_output = self.smpl(**data)
        # smpl_verts = smpl_output.vertices.squeeze().numpy()
        # smpl_verts = data['body_verts'].numpy()
        free_verts = data["body_verts_free"].numpy()
        # smpl_mesh = trimesh.Trimesh(smpl_verts, self.faces, process=False)
        free_verts_mesh = trimesh.Trimesh(free_verts, self.faces, process=False)

        for occ_val in [0, 1, 2]:
            assert np.shape(points[occ == 0])[0] > 0, (
                f"No sampled points with occ value {occ_val}"
            )

        # Print stats
        print(f"Number of points sampled: {points.shape[0]}")
        print(
            f"Number of points with occ value NO: {np.shape(points[occ == 0])[0]}, {np.shape(points[occ == 0])[0] / points.shape[0] * 100:.2f}%"
        )
        print(
            f"Number of points with occ value LT: {np.shape(points[occ == 1])[0]}, {np.shape(points[occ == 1])[0] / points.shape[0] * 100:.2f}%"
        )
        print(
            f"Number of points with occ value AT: {np.shape(points[occ == 2])[0]}, {np.shape(points[occ == 2])[0] / points.shape[0] * 100:.2f}%"
        )
        print(
            f"Number of points with occ value BONE: {np.shape(points[occ == 3])[0]}, {np.shape(points[occ == 3])[0] / points.shape[0] * 100:.2f}%"
        )

        scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0])
        if color_style == "by_class":
            scene.add(vis_create_pc(points[occ == 0], color=(0.0, 0.0, 0.0)))  # NO
            scene.add(vis_create_pc(points[occ == 1], color=(1.0, 0.0, 0.0)))  # LT
            scene.add(vis_create_pc(points[occ == 2], color=(1.0, 1.0, 0.0)))  # AT
            scene.add(vis_create_pc(points[occ == 3], color=(0.0, 0.0, 1.0)))  # BONE
        elif color_style == "by_skinning":
            m = pyrender.Mesh.from_points(points, colors=skinning_color)
            scene.add(m)
        else:
            raise NotImplementedError(f"Color style {color_style} not implemented")
        # scene.add(vis_create_pc(mri_points, color=(0., 0., 0.), radius=0.005))
        # scene.add(pyrender.Mesh.from_trimesh(smpl_mesh, smooth=False, wireframe=True))
        scene.add(
            pyrender.Mesh.from_trimesh(free_verts_mesh, smooth=False, wireframe=True)
        )

        if not can_only:
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        if self.data_cfg["sample_can_points"]:
            scene = pyrender.Scene(
                ambient_light=[0.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0]
            )
            scene.add(vis_create_pc(can_points[can_occ == 0], color=(1.0, 0.0, 0.0)))
            scene.add(vis_create_pc(can_points[can_occ == 1], color=(0.0, 1.0, 0.0)))

            can_vertices = (
                self.smpl.forward_canonical(betas=torch.zeros(1, nb_beta))
                .vertices.detach()
                .cpu()
                .numpy()
            )
            can_mesh = trimesh.Trimesh(
                can_vertices.squeeze(), self.faces, process=False
            )

            scene.add(
                pyrender.Mesh.from_trimesh(can_mesh, smooth=False, wireframe=True)
            )
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

        if self.data_cfg["sample_can_hands"]:
            # hands_can_points=can_pts, hands_can_occ=can_occ
            scene = pyrender.Scene(
                ambient_light=[0.1, 0.1, 0.1], bg_color=[1.0, 1.0, 1.0]
            )
            scene.add(
                vis_create_pc(
                    hands_can_points[hands_can_occ == 0], color=(1.0, 0.0, 0.0)
                )
            )
            scene.add(
                vis_create_pc(
                    hands_can_points[hands_can_occ == 1], color=(0.0, 1.0, 0.0)
                )
            )

            can_vertices = (
                self.smpl.forward_canonical(betas=torch.zeros(1, nb_beta))
                .vertices.detach()
                .cpu()
                .numpy()
            )
            can_mesh = trimesh.Trimesh(
                can_vertices.squeeze(), self.faces, process=False
            )

            scene.add(
                pyrender.Mesh.from_trimesh(can_mesh, smooth=False, wireframe=True)
            )
            pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)

    def find_lowest_b1(self):
        all_beta_1 = [self.smpl_data["betas"][i][1] for i in range(len(self))]
        subj_idx = np.array(all_beta_1).argmin()
        return subj_idx

    def find_highest_b1(self):
        all_beta_1 = [self.smpl_data["betas"][i][1] for i in range(len(self))]
        subj_idx = np.array(all_beta_1).argmax()
        return subj_idx

    @torch.no_grad()
    def _getitem_synthetic(self, return_smpl=False):
        # We add a modulo in get_item so that we can use a batch size bigger than the dataset size.
        # dataset[batch_idx] = dataset[batch_idx % len(dataset)]
        # print(f'idx: {idx}')

        subj_data = {key: self.smpl_data[key][0] for key in self.smpl_data.keys()}

        mri_pose = subj_data["body_pose"].clone()
        mri_global_orient = subj_data["global_orient"].clone()

        # Set all the SMPL params to zero
        subj_data["betas"] = np.zeros_like(subj_data["betas"])
        subj_data["body_pose"] = np.zeros_like(subj_data["body_pose"])
        subj_data["global_orient"] = np.zeros_like(subj_data["global_orient"])
        subj_data["transl"] = np.zeros_like(subj_data["transl"])
        subj_data["global_orient_init"] = np.zeros_like(subj_data["global_orient_init"])

        nb_beta = subj_data["betas"].shape[0]
        nb_pose = subj_data["body_pose"].shape[0]

        xpose = self.smpl.canonical_x_bodypose[0].cpu().numpy()
        random_pose = np.random.randn(nb_pose) * 3.14 / 8
        random_betas = (np.random.rand(nb_beta) - 0.5) * 4
        mri_pose = mri_pose.cpu().numpy()

        if self.data_cfg.synt_style == "random":
            subj_data["body_pose"] = xpose
            subj_data["betas"] = random_betas.astype(np.float32)

        if self.data_cfg.synt_style == "fixed":
            subj_data["body_pose"] = xpose
            subj_data["betas"] = np.zeros(nb_beta).astype(np.float32)
            subj_data["betas"][0:2] = 2

        if self.data_cfg.synt_style == "random_per_joint":
            subj_data["betas"] = random_betas.astype(np.float32)

            subj_data["body_pose"] = np.zeros(nb_pose).astype(np.float32)
            random_param_idx = np.random.randint(0, nb_pose)
            subj_data["body_pose"][random_param_idx] = np.random.randn(1) * 3.14
            # print(subj_data['body_pose'][random_param_idx] )

        # subj_data['body_verts'] = smpl_output.vertices

        # subj_data = {key: val.squeeze(0) if torch.is_tensor(val) else val[0] for key, val in subj_data.items()}  # remove B dim

        # subj_data.update({'v_shaped': v_shaped.squeeze(0)})
        # subj_data.update(self.sample_points(smpl_output, subj_data))

        # if return_smpl:
        #     subj_data.update({'smpl_output': smpl_output})
        return subj_data

    def __len__(self):
        if self.split == "val":
            if self.synthetic:
                return 8
            else:
                return self.smpl_data["betas"].shape[0]
        else:
            if self.synthetic:
                return 128  # Artificial number since the data are generated on the fly
            else:
                return self.smpl_data["betas"].shape[0]


if __name__ == "__main__":

    class Map(dict):
        def __getattr__(self, name):
            return self.get(name)

        def __setattr__(self, name, value):
            self[name] = value

    smpl_cfg = {"gender": "female"}

    data_cfg = Map(
        {
            "synthetic": False,
            "sampling_strategy": "mri",
            "nb_points_canspace": 0,
            "n_skin_pts": 0,
            "sample_can_points": False,
            "sample_can_hands": False,
            "sample_can_toes": False,  # Added
            "n_points_hands": 0,  # Added to fix KeyError
            "n_pts_mri": 15000,
            "uniform_sampling_padding": 0.1,
            "surface_offset_min": 0.01,
            "surface_offset_max": 0.05,
        }
    )

    train_cfg = Map({"to_train": "occ", "comp0_out": False, "mri_values": False})

    print("--- Initializing Dataset Object ---")
    ds = MRIDataset.from_config(
        smpl_cfg=smpl_cfg,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        split="train",
    )

    # SUCCESS: Now we can finally see the bone specialists!
    # visualize_warped_specialist(ds, index=0)
