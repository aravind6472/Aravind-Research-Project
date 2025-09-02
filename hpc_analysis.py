#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import argparse
import mrcfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from Bio.PDB import PDBParser
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import ConcatDataset

# Load MRC file (electron density)
def load_mrc_map(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        density = mrc.data.astype(np.float32)
        voxel_size = np.zeros(3, dtype=np.float32)
        voxel_size[0] = mrc.voxel_size.x
        voxel_size[1] = mrc.voxel_size.y
        voxel_size[2] = mrc.voxel_size.z
    density = (density - np.mean(density)) / (np.std(density) + 1e-5)
    density = np.transpose(density, (2, 1, 0)).astype(np.float16)
    return density, voxel_size

# Convert real coordinates to voxel indices
def voxel_to_coord(coord, voxel_size):
    output = np.zeros(3, dtype=int)
    for i in range(3):
        output[i] = int(round(coord[i] / voxel_size[i]))
    return output

# Generate label map for P atoms
def generate_label_map(pdb_path, density_shape, voxel_size):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("RNA", pdb_path)
    label_map = np.zeros(density_shape, dtype=np.uint8)

    for model in structure:
        for chain in model:
            for residue in chain:
                if 'P' in residue:
                    coord = residue['P'].get_coord()
                    voxel = voxel_to_coord(coord, voxel_size)
                    inside = True
                    for i in range(3):
                        if voxel[i] < 0 or voxel[i] >= density_shape[i]:
                            inside = False
                    if inside:
                        x, y, z = voxel[0], voxel[1], voxel[2]
                        label_map[x, y, z] = 1
    return label_map

# Custom Dataset class
class RNADensityDataset(Dataset): 
    def __init__(self, density, labels, patch_size=32):
        self.density = density.astype(np.float16)
        self.labels = labels.astype(np.float16)
        self.patch_size = patch_size
        self.indices = self._get_patch_indices()

    def _get_patch_indices(self):
        stride = self.patch_size//2
        patch_indices = []

        x = 0
        while x <= self.density.shape[0] - self.patch_size:
            y = 0
            while y <= self.density.shape[1] - self.patch_size:
                z = 0
                while z <= self.density.shape[2] - self.patch_size:
                    patch_indices.append((x, y, z))
                    z += stride
                y += stride
            x += stride

        final_x = self.density.shape[0] - self.patch_size
        final_y = self.density.shape[1] - self.patch_size
        final_z = self.density.shape[2] - self.patch_size
        patch_indices.append((final_x, final_y, final_z))

        return patch_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y, z = self.indices[idx]
        d_patch = self.density[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        l_patch = self.labels[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        return torch.tensor(d_patch[None], dtype=torch.float16), torch.tensor(l_patch[None], dtype=torch.float16)

# 3D CNN model
# class VoxelNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(1, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv3d(32, 1, 1),
#         )

#     def forward(self, x):
#         return self.net(x)

class MultiResBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size = 3, padding = 1)
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size = 5, padding = 2)
        self.conv7 = nn.Conv3d(in_channels, out_channels, kernel_size = 7, padding = 3)
        self.out_conv = nn.Conv3d(3*out_channels, out_channels, kernel_size = 1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        merged = torch.cat([x3, x5, x7], dim=1)
        out = self.out_conv(merged)
        return self.relu(self.bn(out))

class DeepMultiResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = MultiResBlock(1, 32)
        self.block2 = MultiResBlock(32, 64)
        self.block3 = MultiResBlock(64, 64)
        self.block4 = MultiResBlock(64, 64)
        self.final_conv = nn.Conv3d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        skip = x 
        x = self.block3(x) + skip  
        x = self.block4(x)
        return self.final_conv(x)
    
# Training with mixed precision
def train_model(model, dataloader, epochs=20):
    device = torch.device("cuda")
    model = model.to(device)

    pos_weight = torch.tensor([160.0], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                pred = model(x)
                loss = criterion(pred.float(), y.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")

# Main script
def main():

    # parser = argparse.ArgumentParser()
    # parser.add_argument("pos_weight", type = float)
    # args = parser.parse_args()
    # pos_weight = args.pos_weight

    
    # data_pairs = [("/home/aravind6472/highres_maps/7msc_fragment_2.mrc","/home/aravind6472/highres_maps/7msc_fragment_2.pdb" ),
    #               ("/home/aravind6472/highres_maps/7msc_fragment_3.mrc","/home/aravind6472/highres_maps/7msc_fragment_3.pdb"),
    #               ("/home/aravind6472/highres_maps/7msc_fragment.mrc","/home/aravind6472/highres_maps/7msc_fragment.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0081_fragment_2.mrc","/home/aravind6472/highres_maps/emd_0081_fragment_2.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0081_fragment.mrc","/home/aravind6472/highres_maps/emd_0081_fragment.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0081_fragment_3.mrc","/home/aravind6472/highres_maps/emd_0081_fragment_3.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0082_fragment_2.mrc","/home/aravind6472/highres_maps/emd_0082_fragment_2.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0082_fragment.mrc","/home/aravind6472/highres_maps/emd_0082_fragment.pdb"),
    #               ("/home/aravind6472/highres_maps/emd_0098_fragment.mrc","/home/aravind6472/highres_maps/emd_0098_fragment.pdb"),
    #               ("/home/aravind6472/highres_maps/fragment_6gqb.mrc","/home/aravind6472/highres_maps/fragment_6gqb.pdb")]

    data_pairs = [("/home/aravind6472/high_res_mrc/6gqb_highres.mrc","/home/aravind6472/6gqb_processed.pdb"),
                  ("/home/aravind6472/high_res_mrc/7msc_highres.mrc","/home/aravind6472/7msc_processed.pdb"),
                  ("/home/aravind6472/high_res_mrc/3UTR_highres.mrc","/home/aravind6472/3UTR_processed.pdb"),
                  ("/home/aravind6472/high_res_mrc/emd_0082_highres.mrc","/home/aravind6472/emd_0082_processed.pdb"),
                  ("/home/aravind6472/high_res_mrc/emd_0098_highres.mrc","/home/aravind6472/emd_98_processed.pdb")]
    
    all_datasets = []
    for i,(mrc_path, pdb_path) in enumerate(data_pairs):
        density, voxel_size = load_mrc_map(mrc_path)
        label_map = generate_label_map(pdb_path, density.shape, voxel_size)
        dataset = RNADensityDataset(density,label_map,patch_size=32)
        all_datasets.append(dataset)
    
    full_dataset = ConcatDataset(all_datasets)
    train_loader = DataLoader(full_dataset, batch_size = 32, shuffle = True)

    # model = VoxelNet()
    model = DeepMultiResNet()
    train_model(model, train_loader, epochs = 30)

    save_path = "/home/aravind6472/model_weight160.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()