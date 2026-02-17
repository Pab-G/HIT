"""Given a SMPL parameters, infer the tissues occupancy"""

import argparse
import os

import numpy as np
import torch
import trimesh

from hit.utils.data import load_smpl_data
from hit.utils.model import HitLoader
from hit.utils.smpl_utils import get_skinning_weights


def refine_bones_with_fintune(bone_model, bt_mesh, smpl_output, hl_standard,data,  device, res=64):
    v_min = bt_mesh.bounds[0]
    v_max = bt_mesh.bounds[1]

    # Use a standard meshgrid approach to avoid list comprehension scoping issues
    x = np.linspace(v_min[0], v_max[0], res)
    y = np.linspace(v_min[1], v_max[1], res)
    z = np.linspace(v_min[2], v_max[2], res)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')

    # Flatten and stack into Nx3 points
    points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    inside_mask = bt_mesh.contains(points)
    target_points = torch.tensor(points[inside_mask], dtype=torch.float32).to(device)
    weights, _ = get_skinning_weights(
        target_points, 
        smpl_output.vertices[0].detach().cpu().numpy(), 
        hl_standard.smpl
    )
    if not inside_mask.any():
        return {}

    # 3. Batch Query
    weights_torch = torch.from_numpy(weights).float().to(device)
    points_torch = torch.from_numpy(target_points).float().to(device)
    
    batch_size = 50000
    all_class_ids = []

    with torch.no_grad():
        for i in range(0, len(points_torch), batch_size):
            pts_batch = points_torch[i:i+batch_size].unsqueeze(0)
            w_batch = weights_torch[i:i+batch_size].unsqueeze(0)
            
            # Now we call query with the weights we just calculated
            out = bone_model.query(pts_batch, smpl_output, skinning_weights=w_batch)
            
            ids = torch.argmax(out['occ'].squeeze(0), dim=-1).cpu().numpy()
            all_class_ids.append(ids)

    # 4. Reconstruct Grid and Export (Same as before)
    class_ids = np.concatenate(all_class_ids)
    full_labels = np.zeros(len(points))
    full_labels[inside_mask] = class_ids
    grid_3d = full_labels.reshape(res, res, res)
    
    return generate_bone_meshes(grid_3d, v_min, v_max, res)


def main():
    
    parser = argparse.ArgumentParser(description='Infer tissues from SMPL parameters')
    
    parser.add_argument('--to_infer', type=str, default='smpl_template', choices=['smpl_template', 'smpl_file'], 
                        help='Whether to infer from a SMPL template or a SMPL file')
    parser.add_argument('--exp_name', type=str, default='rel_male',
                        help='Name of the checkpoint experiment to use for inference' 
                        ) #TODO change to checkpoint path
    parser.add_argument('--target_body', type=str, default='assets/sit.pkl', 
                        help='Path to the SMPL file to infer tissues from')
    parser.add_argument('--out_folder', type=str, default='output',
                        help='Output folder to save the generated meshes')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--ckpt_choice', type=str, default='last', choices=['best', 'last'],
                        help='Which checkpoint to use for inference')
    parser.add_argument('--output', type=str, default='meshes', choices=['meshes', 'slices'], 
                        help='Form of the output (either extract a mesh of each tissue, either extract slices of each tissue)')
    parser.add_argument('--betas', help="List of the 2 first SMPL betas to use for inference", nargs='+', type=float, default=[0.0, 0.0])
    # to enter this parameter, use the following syntax: --betas 0.0 0.0
    
    args = parser.parse_args()

    exp_name = args.exp_name
    target_body = args.target_body
    ckpt_choice = args.ckpt_choice
    device = torch.device(args.device)
    
    out_folder = os.path.join(args.out_folder, f'{exp_name}_{ckpt_choice}')
    
    # Create a data dictionary containing the SMPL parameters 
    if args.to_infer == 'smpl_template':
        data = {}
        data['global_orient'] = torch.zeros(1, 3).to(device) # Global orientation of the body
        data['body_pose'] = torch.zeros(1, 69).to(device) # Per joint rotation of the body (21 joints x 3 axis)
        data['betas'] = torch.zeros(1, 10).to(device) # Shape parameters, values should be between -2 and 2
        data['betas'][0,0] = args.betas[0]
        data['betas'][0,1] = args.betas[1]
        data['transl'] = torch.zeros(1, 3).to(device) # 3D ranslation of the body in meters
    else:
        assert target_body.endswith('.pkl'), 'target_body should be a pkl file'
        assert os.path.exists(target_body), f'SMPL file "{target_body}" does not exist'
        data = load_smpl_data(target_body, device)

        
    # Create output folder
    os.makedirs(out_folder, exist_ok=True)
    
    # Load HIT model
    hl = HitLoader.load_from_path("/home/yulong/pvbg-thesis/HIT/pretrained/hit_female", "female_hit.ckpt")
    #hl = HitLoader.from_expname(exp_name, ckpt_choice=ckpt_choice)
    hl.load()
    hl.hit_model.apply_compression = False

    # Load fine-tuned model for bone tissue classification
    fine_tuned_model = HitLoader.from_expname('FEMALE_MULTIBONE_2', ckpt_choice='best')
    fine_tuned_model.load()
    fine_tuned_model.hit_model.apply_compression = False
    
    
    # Run smpl forward pass to get the SMPL mesh
    smpl_output = hl.smpl(betas=data['betas'], body_pose=data['body_pose'], global_orient=data['global_orient'], trans=data['transl'])
    
    if args.output == 'slices':
        # Extract slices of each tissue
        # values = ["occ", "sw", "beta"] # Values to extract slices from
        values = ["occ","sw", "beta", "comp"] # Values to extract slices from
        for axis, z0 in zip(['y', 'z', 'z'], [0.0, 0.0, -0.4]):
            images = hl.hit_model.evaluate_slice(data, smpl_output, z0=z0, axis=axis, values=values, res=0.002)
            # Save the slice images
            for value, image in zip(values, images):
                image.save(os.path.join(out_folder, f'{value}_{axis}={z0}.png'))
            print(f'Saved slice {os.path.abspath(out_folder)}')
        print(f'Slices saved in {os.path.abspath(out_folder)}')
        
    elif args.output == 'meshes':

        extracted_bone_meshes = hl.hit_model.forward_rigged_bones(specialist=fine_tuned_model.hit_model, betas=data['betas'], body_pose=data['body_pose'], 
                                                                global_orient=data['global_orient'], 
                                                                transl=data['transl'],
                                                                mise_resolution0=64)        
        # Extract the mesh 
        extracted_meshes, _ = hl.hit_model.forward_rigged(data['betas'], 
                                                                body_pose=data['body_pose'], 
                                                                global_orient=data['global_orient'], 
                                                                transl=data['transl'],
                                                                mise_resolution0=64)

        
        
        
        # Extracted meshes are in the form of a list of 3 trimesh objects corresponding to the 3 tissues 'LT', 'AT', 'BT'
        # LT : Lean Tissue (muscle and organs, merged with the visceral and intra-muscular fat)
        # AT : Adipose Tissue (subcutaneous fat)
        # BT : Bone Tissue (long bones, we only predict the femur, radius-ulna, tibia and fibula)
        
        
        smpl_mesh = trimesh.Trimesh(vertices=smpl_output.vertices[0].detach().cpu().numpy(), faces=hl.smpl.faces)

        # Save all the meshes
        smpl_mesh.export(os.path.join(out_folder, 'smpl_mesh.obj'))
        extracted_meshes[0].export(os.path.join(out_folder, 'LT_mesh.obj'))
        extracted_meshes[1].export(os.path.join(out_folder, 'AT_mesh.obj'))
        extracted_meshes[2].export(os.path.join(out_folder, 'BT_mesh.obj'))

        extracted_bone_meshes[0].export(os.path.join(out_folder, 'Femur_mesh.obj'))
        extracted_bone_meshes[1].export(os.path.join(out_folder, 'Pelvis_mesh.obj'))
        extracted_bone_meshes[2].export(os.path.join(out_folder, 'Humerus_mesh.obj'))
        extracted_bone_meshes[3].export(os.path.join(out_folder, 'Radius_Ulna_mesh.obj'))
        extracted_bone_meshes[4].export(os.path.join(out_folder, 'Tibia_Fibula_mesh.obj'))
        
        print(f'Meshes saved in {os.path.abspath(out_folder)}')
    else:
        raise ValueError(f'Unknown output type {args.output}')


if __name__ == '__main__':
    main()