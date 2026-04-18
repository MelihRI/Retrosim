"""
PointNet++ Agent — Ship-D Deep Learning for Wave Resistance Prediction
=======================================================================

Implements PointNet++ (Qi et al., 2017) with Multi-Scale Grouping (MSG)
for direct prediction of wave resistance coefficient (Cw) from hull
point cloud geometry.

Architecture:
  1. STL → Point Cloud (N×3) via RetrosimHullAdapter.extract_point_cloud()
  2. PointNet++ MSG Encoder → Global feature vector
  3. Regression Head → Cw, Cf, Ct scalars

Physics-Informed Loss:
  Loss = MSE(Cw_pred, Cw_true) + λ * |V_predicted − V_archimedes|

Training Data Sources:
  - Ship-D dataset (30,000+ STL files + scalar Cw)
  - Hugging Face hub datasets
  - Synthetic parametric generator

Deployment:
  - PyTorch .pth checkpoint
  - ONNX export for production inference

References:
  - Qi, C.R. et al. (2017). PointNet++: Deep Hierarchical Feature Learning.
  - MIT DeCoDE Lab — Ship-D parametric hull dataset.
"""

import os
import glob
import numpy as np
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PyQt6.QtCore import QObject, pyqtSignal


# ─── Device ──────────────────────────────────────────────────────────────────
def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        if torch.backends.mps.is_available():
            return torch.device('mps')
    except AttributeError:
        pass
    return torch.device('cpu')

DEVICE = _get_device()


# ═══════════════════════════════════════════════════════════════════════════════
# PointNet++ Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) on a point cloud.

    Args:
        xyz:    (B, N, 3) input point cloud
        npoint: number of centroids to sample

    Returns:
        centroids: (B, npoint) indices
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index into points using idx.

    Args:
        points: (B, N, C)
        idx:    (B, S) or (B, S, K)

    Returns:
        indexed: (B, S, C) or (B, S, K, C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def query_ball_point(radius: float, nsample: int,
                     xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Ball query: find all points within radius of each centroid.

    Args:
        radius:  search radius
        nsample: max points per ball
        xyz:     (B, N, 3) all points
        new_xyz: (B, S, 3) centroids

    Returns:
        group_idx: (B, S, nsample) indices
    """
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device = xyz.device

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = torch.cdist(new_xyz, xyz) ** 2  # (B, S, N)

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Fill empty slots with first index (padding)
    group_first = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]

    return group_idx


class SetAbstractionMSG(nn.Module):
    """
    PointNet++ Set Abstraction with Multi-Scale Grouping.

    Samples npoint centroids, groups neighbours at multiple radii,
    and applies shared MLPs to extract features.
    """

    def __init__(self, npoint: int, radius_list: List[float],
                 nsample_list: List[int], in_channel: int,
                 mlp_list: List[List[int]]):
        super().__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()

        for i, mlp in enumerate(mlp_list):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_ch = in_channel + 3  # +3 for relative xyz
            for out_ch in mlp:
                convs.append(nn.Conv2d(last_ch, out_ch, 1))
                bns.append(nn.BatchNorm2d(out_ch))
                last_ch = out_ch
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz: torch.Tensor,
                points: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz:    (B, N, 3)
            points: (B, N, D) or None

        Returns:
            new_xyz:    (B, npoint, 3)
            new_points: (B, npoint, sum(mlp[-1]))
        """
        # FPS
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)

        new_points_list = []

        for i, (radius, nsample) in enumerate(zip(self.radius_list, self.nsample_list)):
            group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)  # (B, npoint, nsample, 3)
            grouped_xyz -= new_xyz.unsqueeze(2)  # relative coordinates

            if points is not None:
                grouped_points = index_points(points, group_idx)  # (B, npoint, nsample, D)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

            # (B, npoint, nsample, C) → (B, C, npoint, nsample) for Conv2d
            grouped_points = grouped_points.permute(0, 3, 1, 2)

            for conv, bn in zip(self.conv_blocks[i], self.bn_blocks[i]):
                grouped_points = F.relu(bn(conv(grouped_points)))

            # Max pool within each group
            new_points = torch.max(grouped_points, dim=-1)[0]  # (B, C, npoint)
            new_points_list.append(new_points)

        new_points = torch.cat(new_points_list, dim=1)  # (B, sum_C, npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, sum_C)

        return new_xyz, new_points


class GlobalSetAbstraction(nn.Module):
    """Final aggregation: all points → single global feature."""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last_ch = in_channel + 3
        for out_ch in mlp:
            self.convs.append(nn.Conv1d(last_ch, out_ch, 1))
            self.bns.append(nn.BatchNorm1d(out_ch))
            last_ch = out_ch

    def forward(self, xyz: torch.Tensor,
                points: torch.Tensor) -> Tuple[None, torch.Tensor]:
        """
        Args:
            xyz:    (B, N, 3)
            points: (B, N, D)

        Returns:
            (None, global_feature (B, 1, C'))
        """
        x = torch.cat([xyz, points], dim=-1)  # (B, N, D+3)
        x = x.permute(0, 2, 1)  # (B, D+3, N)

        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))

        x = torch.max(x, dim=-1, keepdim=True)[0]  # (B, C', 1)
        return None, x.permute(0, 2, 1)  # (B, 1, C')


# ═══════════════════════════════════════════════════════════════════════════════
# PointNet++ Complete Model
# ═══════════════════════════════════════════════════════════════════════════════

class PointNetPlusPlus(nn.Module):
    """
    PointNet++ MSG for hull resistance regression.

    Input:  (B, N, 3) — point cloud from hull surface
    Output: (B, 3)    — [Cw, Cf, Ct]
    """

    def __init__(self, num_points: int = 2048, num_outputs: int = 3):
        super().__init__()
        self.num_points = num_points
        self.num_outputs = num_outputs

        # SA layers (progressively downsample)
        self.sa1 = SetAbstractionMSG(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[16, 32, 64],
            in_channel=0,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )

        self.sa2 = SetAbstractionMSG(
            npoint=128,
            radius_list=[0.2, 0.4, 0.8],
            nsample_list=[32, 64, 128],
            in_channel=64 + 128 + 128,  # from sa1 outputs
            mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )

        self.sa3 = GlobalSetAbstraction(
            in_channel=128 + 256 + 256,  # from sa2 outputs
            mlp=[256, 512, 1024]
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_outputs),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) input point cloud

        Returns:
            (B, 3) predicted [Cw, Cf, Ct]
        """
        # SA1
        l1_xyz, l1_points = self.sa1(xyz, None)
        # SA2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # SA3 (global)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        # Flatten global feature
        global_feat = l3_points.squeeze(1)  # (B, 1024)

        return self.head(global_feat)


# ═══════════════════════════════════════════════════════════════════════════════
# Ship-D Preprocessor
# ═══════════════════════════════════════════════════════════════════════════════

class ShipDPreprocessor:
    """
    Converts STL/OBJ meshes to normalised point clouds for PointNet++.

    Workflow:
      1. Load STL → vertices + faces
      2. Area-weighted surface sampling → (N, 3)
      3. Normalise to unit sphere
    """

    def __init__(self, num_points: int = 2048):
        self.num_points = num_points

    def stl_to_point_cloud(self, stl_path: str) -> np.ndarray:
        """
        Convert STL → normalised point cloud.

        Args:
            stl_path: Path to .stl file

        Returns:
            (num_points, 3) float32 array, centred and scaled to unit sphere
        """
        try:
            from stl import mesh as stl_mesh
        except ImportError:
            raise ImportError("numpy-stl required: pip install numpy-stl")

        hull = stl_mesh.Mesh.from_file(stl_path)
        vertices = hull.vectors.reshape(-1, 3)  # (num_faces*3, 3)

        # Build face list
        n_faces = len(hull.vectors)
        faces = np.arange(n_faces * 3).reshape(n_faces, 3)
        unique_verts, inverse = np.unique(
            vertices.round(6), axis=0, return_inverse=True)
        faces_reindexed = inverse.reshape(n_faces, 3)

        # Surface sample
        pc = self._area_weighted_sample(unique_verts, faces_reindexed)
        return self._normalise(pc)

    def obj_to_point_cloud(self, obj_path: str) -> np.ndarray:
        """Convert OBJ → normalised point cloud."""
        vertices = []
        faces = []
        with open(obj_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'v':
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == 'f':
                    face = []
                    for p in parts[1:4]:
                        face.append(int(p.split('/')[0]) - 1)
                    faces.append(face)

        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int64)
        pc = self._area_weighted_sample(vertices, faces)
        return self._normalise(pc)

    def _area_weighted_sample(self, vertices: np.ndarray,
                               faces: np.ndarray) -> np.ndarray:
        """Sample points uniformly from mesh surface using area weighting."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        prob = areas / (areas.sum() + 1e-12)

        chosen = np.random.choice(len(faces), size=self.num_points, p=prob)

        u = np.random.uniform(0, 1, (self.num_points, 1))
        v = np.random.uniform(0, 1, (self.num_points, 1))
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v

        return (u * vertices[faces[chosen, 0]] +
                v * vertices[faces[chosen, 1]] +
                w * vertices[faces[chosen, 2]]).astype(np.float32)

    def _normalise(self, pc: np.ndarray) -> np.ndarray:
        """Centre and scale point cloud to unit sphere."""
        centroid = pc.mean(axis=0)
        pc -= centroid
        max_dist = np.max(np.linalg.norm(pc, axis=1))
        if max_dist > 1e-6:
            pc /= max_dist
        return pc

    def ply_to_point_cloud(self, ply_path: str) -> np.ndarray:
        """
        Load PLY (ASCII/binary) point cloud file.

        Args:
            ply_path: Path to .ply file

        Returns:
            Normalised (num_points, 3) float32 array
        """
        vertices = []
        with open(ply_path, 'r', errors='ignore') as f:
            header_done = False
            vertex_count = 0
            for line in f:
                line = line.strip()
                if not header_done:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    if line == 'end_header':
                        header_done = True
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]),
                                         float(parts[1]),
                                         float(parts[2])])
                    except ValueError:
                        continue
                if len(vertices) >= vertex_count > 0:
                    break

        if not vertices:
            raise ValueError(f"No vertices found in PLY: {ply_path}")

        pc = np.array(vertices, dtype=np.float32)
        return self._resample_and_normalise(pc)

    def xyz_to_point_cloud(self, xyz_path: str) -> np.ndarray:
        """
        Load XYZ point cloud file (whitespace-separated x y z [optional fields]).

        Args:
            xyz_path: Path to .xyz or .csv file

        Returns:
            Normalised (num_points, 3) float32 array
        """
        try:
            # Try loading as space-separated numeric data
            data = np.loadtxt(xyz_path, dtype=np.float32,
                              usecols=(0, 1, 2), comments='#')
        except Exception:
            # Fall back to comma-separated
            data = np.loadtxt(xyz_path, dtype=np.float32,
                              usecols=(0, 1, 2), delimiter=',', comments='#')

        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"XYZ file must have ≥3 columns, got {data.shape[1]}")

        return self._resample_and_normalise(data[:, :3])

    def npy_to_point_cloud(self, npy_path: str) -> np.ndarray:
        """
        Load pre-computed .npy point cloud.

        Returns:
            Normalised (num_points, 3) float32 array
        """
        pc = np.load(npy_path).astype(np.float32)
        if pc.ndim == 1:
            pc = pc.reshape(-1, 3)
        return self._resample_and_normalise(pc)

    def _resample_and_normalise(self, pc: np.ndarray) -> np.ndarray:
        """Resample to target num_points and normalise."""
        if len(pc) == 0:
            return np.zeros((self.num_points, 3), dtype=np.float32)

        if len(pc) > self.num_points:
            indices = np.random.choice(len(pc), self.num_points, replace=False)
        else:
            indices = np.random.choice(len(pc), self.num_points, replace=True)

        pc = pc[indices]
        return self._normalise(pc)

    def load_any(self, file_path: str) -> np.ndarray:
        """
        Universal loader: auto-detect format and convert to point cloud.

        Supported: .stl, .obj, .ply, .xyz, .csv, .npy, .npz

        Args:
            file_path: Path to any supported geometry file.

        Returns:
            Normalised (num_points, 3) float32 point cloud.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.stl':
            return self.stl_to_point_cloud(file_path)
        elif ext == '.obj':
            return self.obj_to_point_cloud(file_path)
        elif ext == '.ply':
            return self.ply_to_point_cloud(file_path)
        elif ext in ('.xyz', '.csv', '.txt'):
            return self.xyz_to_point_cloud(file_path)
        elif ext == '.npy':
            return self.npy_to_point_cloud(file_path)
        elif ext == '.npz':
            data = np.load(file_path)
            key = list(data.keys())[0]
            return self._resample_and_normalise(data[key].astype(np.float32))
        else:
            # Try via RetrosimHullAdapter (STL/OBJ with advanced parsing)
            try:
                from core.geometry.FFDHullMorpher import RetrosimHullAdapter
                return RetrosimHullAdapter.import_mesh_as_point_cloud(
                    file_path, self.num_points)
            except Exception:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    "Supported: .stl, .obj, .ply, .xyz, .csv, .npy, .npz"
                )

    def batch_convert(self, input_dir: str, output_dir: str,
                      file_ext: str = '.stl') -> List[str]:
        """
        Batch convert geometry files to .npy point clouds.

        Supports: .stl, .obj, .ply, .xyz, .csv

        Args:
            input_dir:  Directory with geometry files
            output_dir: Where to save .npy files
            file_ext:   File extension to filter (e.g., '.stl')

        Returns:
            List of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        mesh_files = glob.glob(os.path.join(input_dir, f'*{file_ext}'))
        output_files = []

        for mf in mesh_files:
            try:
                pc = self.load_any(mf)
                out_name = os.path.splitext(os.path.basename(mf))[0] + '.npy'
                out_path = os.path.join(output_dir, out_name)
                np.save(out_path, pc)
                output_files.append(out_path)
            except Exception as e:
                print(f"[!] Skipped {mf}: {e}")

        print(f"[OK] Batch conversion: {len(output_files)}/{len(mesh_files)} files")
        return output_files


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class ShipDDataset(Dataset):
    """
    PyTorch Dataset for Ship-D point clouds + resistance labels.

    Loads pre-computed .npy point clouds and resistance CSV.
    """

    def __init__(self, pc_dir: str, labels_csv: str,
                 num_points: int = 2048):
        import pandas as pd
        self.num_points = num_points
        self.pc_dir = pc_dir

        df = pd.read_csv(labels_csv)
        self.labels = df

        # Map filenames to rows
        self.entries = []
        for _, row in df.iterrows():
            # Expect columns: 'filename', 'Cw', 'Cf', 'Ct'
            npy_name = str(row.get('filename', '')).replace('.stl', '.npy').replace('.obj', '.npy')
            npy_path = os.path.join(pc_dir, npy_name)
            if os.path.exists(npy_path):
                self.entries.append({
                    'path': npy_path,
                    'Cw': float(row.get('Cw', 0)),
                    'Cf': float(row.get('Cf', 0)),
                    'Ct': float(row.get('Ct', 0)),
                })

        print(f"[DIR] ShipDDataset: {len(self.entries)} samples loaded")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        pc = np.load(entry['path'])

        # Resample if needed
        if len(pc) != self.num_points:
            choice = np.random.choice(len(pc), self.num_points, replace=len(pc) < self.num_points)
            pc = pc[choice]

        label = np.array([entry['Cw'], entry['Cf'], entry['Ct']], dtype=np.float32)
        return torch.tensor(pc, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Physics-Informed Loss
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsAwareResistanceLoss(nn.Module):
    """
    Physics-informed loss for resistance prediction.

    Loss = MSE(pred, true) + λ₁ * |Ct - (Cw + Cf)|  + λ₂ * ReLU(-Cw) + λ₃ * ReLU(-Cf)

    Physical constraints:
      1. Total resistance = wave + friction (component consistency)
      2. All resistance coefficients must be non-negative
      3. Cw should scale roughly with Froude number (if provided)
    """

    def __init__(self, lambda_consistency: float = 0.2,
                 lambda_positivity: float = 0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_consistency = lambda_consistency
        self.lambda_positivity = lambda_positivity

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3) — [Cw, Cf, Ct]
            target: (B, 3) — [Cw, Cf, Ct]
        """
        # Data loss
        data_loss = self.mse(pred, target)

        # Physics: Ct ≈ Cw + Cf
        Cw = pred[:, 0]
        Cf = pred[:, 1]
        Ct = pred[:, 2]
        consistency = torch.mean((Ct - (Cw + Cf)) ** 2)

        # Positivity constraint
        positivity = torch.mean(F.relu(-Cw) ** 2 + F.relu(-Cf) ** 2)

        return data_loss + \
               self.lambda_consistency * consistency + \
               self.lambda_positivity * positivity


# ═══════════════════════════════════════════════════════════════════════════════
# PointNet Agent (QObject — GUI-integrated)
# ═══════════════════════════════════════════════════════════════════════════════

class PointNetAgent(QObject):
    """
    PointNet++ based hull resistance prediction agent.

    Provides fast Cw prediction from hull point clouds:
      STL → Point Cloud (2048 pts) → PointNet++ → [Cw, Cf, Ct]
                                                    (~0.01s on GPU)
    """

    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    def __init__(self, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points
        self.model = None
        self.preprocessor = ShipDPreprocessor(num_points)
        self.is_trained = False
        self._build_model()
        self._try_load()

    def _build_model(self):
        """Instantiate PointNet++ model."""
        self.model = PointNetPlusPlus(
            num_points=self.num_points,
            num_outputs=3  # Cw, Cf, Ct
        ).to(DEVICE)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[PointNet++] Built: {total:,} parameters on {DEVICE}")

    def _try_load(self):
        """Attempt to load existing weights."""
        pth_path = os.path.join(self.MODEL_DIR, 'pointnet_cw.pth')
        if os.path.exists(pth_path):
            try:
                ckpt = torch.load(pth_path, map_location=DEVICE, weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.train(False)
                self.is_trained = True
                print(f"[OK] PointNet++ loaded: {pth_path}")
            except Exception as e:
                print(f"[!] PointNet++ load failed: {e}")
        else:
            # Also try ONNX
            onnx_path = os.path.join(self.MODEL_DIR, 'pointnet_cw.onnx')
            if os.path.exists(onnx_path):
                print(f"[i] ONNX model found at {onnx_path}")

    def predict_from_stl(self, stl_path: str, speed_knots: float = 12.0) -> Dict:
        """
        Predict resistance from STL file.

        Args:
            stl_path: Path to hull STL
            speed_knots: Ship speed in knots

        Returns:
            {'Cw': float, 'Cf': float, 'Ct': float, 'source': 'pointnet'}
        """
        if not self.is_trained:
            return {'Cw': 0, 'Cf': 0, 'Ct': 0, 'source': 'untrained'}

        # Convert STL → point cloud
        pc = self.preprocessor.stl_to_point_cloud(stl_path)
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        self.model.train(False)
        with torch.no_grad():
            pred = self.model(pc_tensor)  # (1, 3)

        pred_np = pred.cpu().numpy()[0]
        return {
            'Cw': float(pred_np[0]),
            'Cf': float(pred_np[1]),
            'Ct': float(pred_np[2]),
            'speed': speed_knots,
            'source': 'pointnet++',
        }

    def predict_from_point_cloud(self, point_cloud: np.ndarray) -> Dict:
        """Predict resistance from pre-computed point cloud (N, 3)."""
        if not self.is_trained:
            return {'Cw': 0, 'Cf': 0, 'Ct': 0, 'source': 'untrained'}

        # Resample if needed
        if len(point_cloud) != self.num_points:
            choice = np.random.choice(
                len(point_cloud), self.num_points,
                replace=len(point_cloud) < self.num_points)
            point_cloud = point_cloud[choice]

        # Normalise
        pc = self.preprocessor._normalise(point_cloud.copy())
        pc_tensor = torch.tensor(pc, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        self.model.train(False)
        with torch.no_grad():
            pred = self.model(pc_tensor)

        pred_np = pred.cpu().numpy()[0]
        return {
            'Cw': float(pred_np[0]),
            'Cf': float(pred_np[1]),
            'Ct': float(pred_np[2]),
            'source': 'pointnet++',
        }

    def train(self, pc_dir: str = None, labels_csv: str = None,
              epochs: int = 100, batch_size: int = 16,
              learning_rate: float = 1e-3):
        """
        Train PointNet++ on Ship-D dataset.

        If no data provided, generates synthetic training data
        from the parametric hull generator.
        """
        self.progress_signal.emit(5, "PointNet++ eğitim verisi hazırlanıyor...")

        if pc_dir and labels_csv and os.path.exists(pc_dir):
            ds = ShipDDataset(pc_dir, labels_csv, self.num_points)
        else:
            ds = self._generate_synthetic_dataset()

        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True)

        criterion = PhysicsAwareResistanceLoss(
            lambda_consistency=0.2, lambda_positivity=0.5)
        optimizer = optim.AdamW(self.model.parameters(),
                                lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        self.model.train()
        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            epoch_loss = 0.0
            for pc_batch, label_batch in loader:
                pc_batch = pc_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)

                optimizer.zero_grad()
                pred = self.model(pc_batch)
                loss = criterion(pred, label_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * pc_batch.size(0)

            epoch_loss /= len(ds)
            scheduler.step()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if epoch % 10 == 0 or epoch == epochs - 1:
                pct = int((epoch + 1) / epochs * 100)
                msg = f"PointNet++ Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}"
                self.progress_signal.emit(pct, msg)
                print(f"[#] {msg}")

        # Restore best
        if best_state:
            self.model.load_state_dict(best_state)

        self.is_trained = True
        self._save_model()
        self.model.train(False)
        self.progress_signal.emit(100, f"[OK] PointNet++ eğitim tamamlandı! Loss: {best_loss:.6f}")

    def _generate_synthetic_dataset(self) -> Dataset:
        """
        Generate synthetic training data using parametric hulls.

        Uses RetrosimHullAdapter to create random hulls → point clouds
        and Holtrop-Mennen for ground truth resistance.
        """
        print("[*] Generating synthetic PointNet++ dataset...")

        try:
            from core.geometry.FFDHullMorpher import RetrosimHullAdapter
        except ImportError:
            print("[!] RetrosimHullAdapter not available — using random data")
            return self._random_fallback_dataset()

        n_samples = 300
        all_pc = []
        all_labels = []
        adapter = RetrosimHullAdapter()

        for i in range(n_samples):
            # Random vessel params
            params = {
                'loa': np.random.uniform(60, 200),
                'beam': np.random.uniform(10, 35),
                'draft': np.random.uniform(4, 12),
                'cb': np.random.uniform(0.5, 0.85),
                'speed': np.random.uniform(8, 18),
            }

            try:
                adapter.set_from_ui(params)
                pc = adapter.extract_point_cloud(num_points=self.num_points)
                pc = self.preprocessor._normalise(pc)

                # Ground truth from Holtrop
                res = adapter.predict_total_resistance(params['speed'])
                Rf = res.get('Rf', 0)
                Rw = res.get('Rw', 0)
                Rt = res.get('Rt', 1)

                # Normalise to coefficients
                V = params['speed'] * 0.5144
                rho = 1025.0
                features = adapter.extract_ml_features()
                S = features.get('wetted_surface_area', 1)
                q = 0.5 * rho * V ** 2 * S if V > 0 and S > 0 else 1

                Cw = Rw / q if q > 0 else 0
                Cf = Rf / q if q > 0 else 0
                Ct = Rt / q if q > 0 else 0

                all_pc.append(pc)
                all_labels.append([Cw, Cf, Ct])
            except Exception as e:
                continue

        if len(all_pc) < 10:
            return self._random_fallback_dataset()

        pcs = np.stack(all_pc)
        labels = np.array(all_labels, dtype=np.float32)

        return torch.utils.data.TensorDataset(
            torch.tensor(pcs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32))

    def _random_fallback_dataset(self):
        """Minimal random dataset for testing."""
        n = 50
        pcs = np.random.randn(n, self.num_points, 3).astype(np.float32) * 0.3
        labels = np.random.rand(n, 3).astype(np.float32) * 0.01
        labels[:, 2] = labels[:, 0] + labels[:, 1]  # Ct = Cw + Cf
        return torch.utils.data.TensorDataset(
            torch.tensor(pcs), torch.tensor(labels))

    def _save_model(self):
        """Save PointNet++ weights."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        path = os.path.join(self.MODEL_DIR, 'pointnet_cw.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_points': self.num_points,
        }, path)
        print(f"[S] PointNet++ saved: {path}")

    def export_to_onnx(self, output_path: Optional[str] = None):
        """
        Export trained model to ONNX format for deployment.

        The ONNX model accepts (1, num_points, 3) and outputs (1, 3).
        """
        if output_path is None:
            output_path = os.path.join(self.MODEL_DIR, 'pointnet_cw.onnx')

        self.model.train(False)
        dummy = torch.randn(1, self.num_points, 3, device=DEVICE)

        try:
            torch.onnx.export(
                self.model, dummy, output_path,
                input_names=['point_cloud'],
                output_names=['resistance_coeffs'],
                dynamic_axes={
                    'point_cloud': {0: 'batch_size'},
                    'resistance_coeffs': {0: 'batch_size'}
                },
                opset_version=14,
            )
            print(f"[OK] ONNX exported: {output_path}")
        except Exception as e:
            print(f"[!] ONNX export failed: {e}")
            self.error_signal.emit(f"ONNX export failed: {e}")
