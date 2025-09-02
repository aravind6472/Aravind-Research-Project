#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import mrcfile
import torch
import torch.nn as nn
from Bio.PDB import PDBIO, StructureBuilder
from torch.cuda.amp import autocast
from Bio.PDB import MMCIFIO

class VoxelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 1, 1),
        )

    def forward(self, x):
        return self.net(x)

# --- MRC loading and preprocessing ---
def load_unseen_mrc(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        density = mrc.data.astype(np.float32)
        voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32)
    density = (density - np.mean(density)) / (np.std(density) + 1e-5)
    density = np.transpose(density, (2, 1, 0))  # Shape: (X, Y, Z)
    return density, voxel_size

# --- Convert voxel index to coordinate ---
def coord_from_voxel(voxel, voxel_size):
    coord = np.zeros(3, dtype=np.float32)
    for i in range(3):
        coord[i] = voxel[i] * voxel_size[i]
    return coord

# --- Inference on the full volume with sliding patches ---
def predict_p_atoms(model, density, voxel_size, patch_size=32, threshold=0.5):
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    stride = patch_size // 2
    pred_coords = []

    x = 0
    while x <= density.shape[0] - patch_size:
        y = 0
        while y <= density.shape[1] - patch_size:
            z = 0
            while z <= density.shape[2] - patch_size:
                patch = density[x:x+patch_size, y:y+patch_size, z:z+patch_size]
                patch_tensor = torch.tensor(patch[None, None], dtype=torch.float16).to(device)

                with autocast():
                    output = model(patch_tensor).sigmoid().detach().cpu().numpy()[0, 0]

                for i in range(patch_size):
                    for j in range(patch_size):
                        for k in range(patch_size):
                            if output[i, j, k] > threshold:
                                coord = coord_from_voxel((x+i, y+j, z+k), voxel_size)
                                pred_coords.append(coord)

                z += stride
            y += stride
        x += stride

    return pred_coords

# --- Write coordinates to a PDB file ---
def write_cif(pred_coords, cif_path):
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure("PRED")
    builder.init_model(0)
    builder.init_chain("A")

    for i,coord in enumerate(pred_coords):
        builder.init_seg("    ")
        builder.init_residue("P", " ", i, " ")
        builder.init_atom("P", coord, 1.0, 1.0, " ", "P", i, element="P")
    
    structure = builder.get_structure()
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_path)

# --- Main entry point ---
def run_inference():
    mrc_path = "/home/aravind6472/3UTR_all-original.mrc"
    model_path = "/home/aravind6472/trained_model.pt"
    output_cif_path = "/home/aravind6472/predicted_P_atoms.cif"
    patch_size = 32
    threshold = 0.6

    print("Loading MRC file...")
    density, voxel_size = load_unseen_mrc(mrc_path)

    print("Loading trained model...")
    model = VoxelNet()
    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    print("Running inference...")
    pred_coords = predict_p_atoms(model, density, voxel_size, patch_size, threshold)

    print(f"Writing {len(pred_coords)} predicted P atoms to: {output_cif_path}")
    write_cif(pred_coords, output_cif_path)
    print("Done.")

if __name__ == "__main__":
    run_inference()

