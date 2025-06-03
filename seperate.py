import os
import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from plyfile import PlyData, PlyElement
import argparse
import json

def save_selected_gaussians(gaussians, selected_indices, path):
    """Save a subset of Gaussians to a .ply file based on selected indices."""
    selected_indices = selected_indices.cpu().numpy()
    xyz = gaussians._xyz[selected_indices].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc[selected_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest[selected_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity[selected_indices].detach().cpu().numpy()
    scale = gaussians._scaling[selected_indices].detach().cpu().numpy()
    rotation = gaussians._rotation[selected_indices].detach().cpu().numpy()
    obj_dc = gaussians._objects_dc[selected_indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, obj_dc), axis=1)
    dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def main():
    parser = argparse.ArgumentParser(description="Separate Gaussians into individual .ply files based on object indices.")
    parser.add_argument("--associated_mask_folder", type=str, required=True, help="Path to the associated mask folder containing gaussian_idx_bank.pt and info.json")
    parser.add_argument("--sh_degree", type=int, default=3, help="Spherical harmonics degree for the Gaussian model")
    args = parser.parse_args()

    # Paths
    associated_mask_folder = args.associated_mask_folder
    info_path = os.path.join(associated_mask_folder, "info.json")
    gaussian_idx_bank_path = os.path.join(associated_mask_folder, "gaussian_idx_bank.pt")
    output_folder = os.path.join(associated_mask_folder, "separated_objects")

    # Load metadata
    with open(info_path, "r") as f:
        info = json.load(f)
    ply_path = info["ply_path"]

    # Load Gaussian index bank
    gaussian_idx_bank = torch.load(gaussian_idx_bank_path)

    # Load Gaussian model
    gaussians = GaussianModel(sh_degree=args.sh_degree)
    gaussians.load_ply(ply_path)

    # Create output directory
    os.makedirs(output_folder, exist_ok=True)

    # Separate and save each object's Gaussians
    for i, indices in enumerate(gaussian_idx_bank):
        if len(indices) > 0:  # Only save if there are Gaussians for this object
            save_path = os.path.join(output_folder, f"object_{i}.ply")
            save_selected_gaussians(gaussians, indices, save_path)
            print(f"Saved object {i} with {len(indices)} Gaussians to {save_path}")

if __name__ == "__main__":
    main()