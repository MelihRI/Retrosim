"""
Retrosim Pipeline Orchestrator
===============================
End-to-end pipeline: Design Vector -> Geometry -> CFD -> FNO Training -> Inference.

Connects:
  core.geometry_assembler.GeometryAssembler  (hull STL + SDF + input tensor)
  core.openfoam_runner.OpenFOAMRunner        (ground-truth flow fields)
  models.gc_fno3d.GC_FNO3D                  (surrogate model)
  models.gc_fno3d_loss.GC_FNO3DLoss         (composite loss)

License: Apache 2.0
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.geometry_assembler import GeometryAssembler
from core.openfoam_runner import OpenFOAMRunner
from models.gc_fno3d import GC_FNO3D
from models.gc_fno3d_loss import GC_FNO3DLoss


class RetrosimPipeline:
    """Orchestrates the 3-stage Retrosim workflow.

    Stage 1 -- Dataset generation:
        design_vector -> GeometryAssembler -> STL + SDF + input_tensor
                      -> OpenFOAMRunner    -> flow_field + C_T

    Stage 2 -- Training:
        (input_tensor, flow_field, C_T) -> GC_FNO3D + GC_FNO3DLoss

    Stage 3 -- Inference:
        design_vector -> GeometryAssembler -> GC_FNO3D -> (flow, C_T)
    """

    def __init__(
        self,
        assembler: GeometryAssembler,
        runner: OpenFOAMRunner,
        model: GC_FNO3D,
        loss_fn: GC_FNO3DLoss,
        device: str = "cuda",
    ):
        self.assembler = assembler
        self.runner = runner
        self.model = model
        self.loss_fn = loss_fn
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        print(f"[Pipeline] Device: {self.device} | "
              f"Params: {self.model.count_parameters():,}")

    # ================================================================== #
    # Stage 1: Dataset Generation                                         #
    # ================================================================== #

    def generate_dataset(
        self,
        design_vectors: np.ndarray,
        appendage_configs: List[dict],
        save_dir: Optional[Path] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate paired (input, ground-truth) data via assembler + CFD.

        Parameters
        ----------
        design_vectors : (N, 45) array
        appendage_configs : list of N dicts, each with keys:
            "stl_path": Path|None, "transform": (4,4)|None,
            "Re": float, "Fr": float, "U_inf": float,
            "operating_param": float (optional, default 1.0)
        save_dir : optional directory to persist as dataset.npz

        Returns
        -------
        X      : (N, 6, D, H, W) float32 tensor
        Y_flow : (N, 4, D, H, W) float32 tensor
        Y_ct   : (N, 1) float32 tensor
        """
        N = len(design_vectors)
        assert len(appendage_configs) == N, (
            f"design_vectors ({N}) and appendage_configs ({len(appendage_configs)}) "
            f"must have the same length."
        )

        X_list, Y_flow_list, Y_ct_list = [], [], []
        t0 = time.time()

        for i in range(N):
            dv = design_vectors[i]
            cfg = appendage_configs[i]
            Re = cfg["Re"]
            Fr = cfg["Fr"]
            U_inf = cfg["U_inf"]

            print(f"\n[Pipeline] Sample {i+1}/{N}  Re={Re:.2e} Fr={Fr:.4f}")

            # --- Stage 1a: Geometry assembly ---
            geo = self.assembler.build(
                design_vector=dv,
                appendage_stl=cfg.get("stl_path"),
                appendage_transform=cfg.get("transform"),
                Re=Re, Fr=Fr,
                operating_param=cfg.get("operating_param", 1.0),
            )
            input_tensor = geo["input_tensor"]   # (6, D, H, W)
            sdf = geo["sdf"]                     # (1, D, H, W)
            stl_obj = geo["combined_stl"]

            # Save STL to temp for OpenFOAM
            if save_dir:
                stl_dir = Path(save_dir) / "stl"
                stl_dir.mkdir(parents=True, exist_ok=True)
                stl_path = stl_dir / f"hull_{i:04d}.stl"
            else:
                import tempfile
                stl_path = Path(tempfile.mkdtemp()) / f"hull_{i:04d}.stl"

            stl_obj.save(str(stl_path))

            # --- Stage 1b: OpenFOAM ground truth ---
            case_dir = (Path(save_dir) / f"case_{i:04d}") if save_dir else (
                Path(stl_path).parent / f"case_{i:04d}")
            try:
                cfd = self.runner.run_case(
                    combined_stl_path=stl_path,
                    Re=Re, Fr=Fr, U_inf=U_inf,
                    case_dir=case_dir,
                )
                flow_field = cfd["flow_field"]   # (4, D, H, W)
                C_T = cfd["C_T"]
            except RuntimeError as e:
                print(f"[Pipeline] WARN: CFD failed for sample {i}: {e}")
                print("[Pipeline]        Skipping this sample.")
                continue

            X_list.append(input_tensor)
            Y_flow_list.append(flow_field)
            Y_ct_list.append(C_T)

        if not X_list:
            raise RuntimeError("No samples generated successfully.")

        X = torch.tensor(np.stack(X_list), dtype=torch.float32)
        Y_flow = torch.tensor(np.stack(Y_flow_list), dtype=torch.float32)
        Y_ct = torch.tensor(Y_ct_list, dtype=torch.float32).unsqueeze(-1)

        elapsed = time.time() - t0
        print(f"\n[Pipeline] Dataset: {len(X_list)}/{N} samples in {elapsed:.0f}s")
        print(f"           X: {list(X.shape)}  Y_flow: {list(Y_flow.shape)}  "
              f"Y_ct: {list(Y_ct.shape)}")

        # Persist
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            npz_path = save_dir / "dataset.npz"
            np.savez_compressed(
                str(npz_path),
                X=X.numpy(), Y_flow=Y_flow.numpy(), Y_ct=Y_ct.numpy(),
            )
            print(f"[Pipeline] Saved: {npz_path}")

        return X, Y_flow, Y_ct

    # ================================================================== #
    # Stage 2: Training                                                   #
    # ================================================================== #

    def train(
        self,
        X: torch.Tensor,
        Y_flow: torch.Tensor,
        Y_ct: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 4,
        grad_clip: float = 1.0,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_every: int = 25,
    ) -> List[dict]:
        """Train the GC-FNO3D surrogate on the generated dataset.

        Parameters
        ----------
        X      : (N, 6, D, H, W)
        Y_flow : (N, 4, D, H, W)
        Y_ct   : (N, 1)
        epochs, lr, batch_size : training hyperparameters
        grad_clip : max gradient norm (0 to disable)
        checkpoint_dir : if set, saves model every checkpoint_every epochs

        Returns
        -------
        list[dict] -- per-epoch metrics
        """
        N = X.shape[0]
        print(f"\n[Train] Starting: {N} samples, {epochs} epochs, "
              f"bs={batch_size}, lr={lr}")

        dataset = TensorDataset(X, Y_flow, Y_ct)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            pin_memory=(self.device.type == "cuda"),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )

        self.model.train()
        history: List[dict] = []
        best_loss = float("inf")
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_data = 0.0
            epoch_ct_mae = 0.0
            n_batches = 0

            for x_batch, y_flow_batch, y_ct_batch in loader:
                x_batch = x_batch.to(self.device)
                y_flow_batch = y_flow_batch.to(self.device)
                y_ct_batch = y_ct_batch.to(self.device)

                # SDF is channel 0 of input
                sdf_batch = x_batch[:, 0:1, :, :, :]

                # Forward
                pred = self.model(x_batch, sdf_batch)
                pred_flow = pred["field"]    # (B, 4, D, H, W)
                pred_ct = pred["C_T"]        # (B, 1)

                # Loss
                loss_dict = self.loss_fn(
                    pred,
                    gt_field=y_flow_batch,
                    gt_ct=y_ct_batch,
                    sdf=sdf_batch,
                )
                loss = loss_dict["total"]

                # Backward
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip,
                    )
                optimizer.step()

                # Metrics
                batch_ct_mae = (pred_ct - y_ct_batch).abs().mean().item()
                epoch_loss += loss.item()
                epoch_data += loss_dict.get("data_field", 0.0)
                epoch_ct_mae += batch_ct_mae
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_data = epoch_data / max(n_batches, 1)
            avg_mae = epoch_ct_mae / max(n_batches, 1)
            cur_lr = optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch,
                "loss_total": avg_loss,
                "loss_data": avg_data,
                "ct_mae": avg_mae,
                "lr": cur_lr,
            }
            history.append(metrics)

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Logging
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:4d}/{epochs} | "
                    f"Loss: {avg_loss:.6f} | "
                    f"Data: {avg_data:.6f} | "
                    f"C_T MAE: {avg_mae:.6f} | "
                    f"LR: {cur_lr:.1e} | "
                    f"{elapsed:.0f}s"
                )

            # Checkpoint
            if checkpoint_dir and epoch % checkpoint_every == 0:
                self._save_checkpoint(checkpoint_dir, epoch, optimizer, best_loss)

        elapsed = time.time() - t0
        print(f"\n[Train] Complete: {elapsed:.1f}s | Best loss: {best_loss:.6f}")

        # Final checkpoint
        if checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, epochs, optimizer, best_loss)

        return history

    # ================================================================== #
    # Stage 3: Inference                                                  #
    # ================================================================== #

    @torch.no_grad()
    def predict(
        self,
        design_vector: np.ndarray,
        appendage_config: dict,
    ) -> dict:
        """Run inference on a single hull (no OpenFOAM).

        Parameters
        ----------
        design_vector : (45,) array
        appendage_config : dict with Re, Fr, stl_path, transform, etc.

        Returns
        -------
        dict with:
            "flow_field" : (4, D, H, W) numpy
            "C_T"        : float
            "sdf"        : (1, D, H, W) numpy
        """
        self.model.eval()

        geo = self.assembler.build(
            design_vector=design_vector,
            appendage_stl=appendage_config.get("stl_path"),
            appendage_transform=appendage_config.get("transform"),
            Re=appendage_config["Re"],
            Fr=appendage_config["Fr"],
            operating_param=appendage_config.get("operating_param", 1.0),
        )

        x = torch.tensor(geo["input_tensor"], dtype=torch.float32)
        x = x.unsqueeze(0).to(self.device)           # (1, 6, D, H, W)
        sdf = x[:, 0:1, :, :, :]                     # (1, 1, D, H, W)

        pred = self.model(x, sdf)

        return {
            "flow_field": pred["field"][0].cpu().numpy(),     # (4, D, H, W)
            "C_T": pred["C_T"][0, 0].cpu().item(),            # float
            "sdf": geo["sdf"],                                # (1, D, H, W)
        }

    # ================================================================== #
    # Batch Inference                                                     #
    # ================================================================== #

    @torch.no_grad()
    def predict_batch(
        self,
        design_vectors: np.ndarray,
        appendage_configs: List[dict],
        batch_size: int = 8,
    ) -> List[dict]:
        """Run inference on multiple hulls without OpenFOAM.

        Returns list of dicts (same schema as predict()).
        """
        self.model.eval()
        results = []

        for i in range(len(design_vectors)):
            result = self.predict(design_vectors[i], appendage_configs[i])
            results.append(result)

        return results

    # ================================================================== #
    # Save / Load                                                         #
    # ================================================================== #

    def _save_checkpoint(
        self, ckpt_dir: Path, epoch: int,
        optimizer: torch.optim.Optimizer, best_loss: float,
    ):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"gc_fno3d_epoch{epoch:04d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }, path)
        print(f"  [Checkpoint] {path.name}")

    def save_model(self, path: Path):
        """Save trained model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[Pipeline] Model saved: {path}")

    def load_model(self, path: Path):
        """Load trained model weights."""
        path = Path(path)
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"[Pipeline] Model loaded: {path}")

    @staticmethod
    def load_dataset(path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load a dataset.npz previously saved by generate_dataset()."""
        data = np.load(str(path))
        X = torch.tensor(data["X"], dtype=torch.float32)
        Y_flow = torch.tensor(data["Y_flow"], dtype=torch.float32)
        Y_ct = torch.tensor(data["Y_ct"], dtype=torch.float32)
        print(f"[Pipeline] Loaded dataset: X={list(X.shape)} "
              f"Y_flow={list(Y_flow.shape)} Y_ct={list(Y_ct.shape)}")
        return X, Y_flow, Y_ct
