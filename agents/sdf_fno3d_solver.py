"""
SDF-Conditioned 3D-FNO Navier-Stokes PINN Solver — Training & Entry Point
==========================================================================
Pure PINN training (no ground-truth data required).
Assembles SDFGenerator + FNO3d_NS_Solver + NavierStokesPINNLoss.

Usage:
    python agents/sdf_fno3d_solver.py                      # analytical hull, CPU
    python agents/sdf_fno3d_solver.py --stl hull.stl       # real hull STL
    python agents/sdf_fno3d_solver.py --gpu --epochs 1000  # GPU training
"""

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import torch
import torch.optim as optim

from agents.sdf_utils import SolverConfig, SDFGenerator
from agents.fno3d_network import (
    FNO3d_NS_Solver,
    NavierStokesPINNLoss,
    DEVICE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Engine
# ═══════════════════════════════════════════════════════════════════════════════

class PINNTrainer:
    """Trains FNO3d_NS_Solver in pure physics-informed mode (no labelled data).

    The SDF is precomputed once and cached. Each epoch evaluates the PDE
    residuals on the full grid and backpropagates through the network.
    """

    def __init__(self, config: SolverConfig, stl_path: Optional[str] = None):
        self.config = config
        self.device = DEVICE if torch.cuda.is_available() or config.use_amp else torch.device('cpu')

        print(f"{'='*60}")
        print(f" SDF-Conditioned 3D-FNO PINN Solver")
        print(f" Grid: {config.grid_dims}  |  Device: {self.device}")
        print(f" Re={config.reynolds:.0e}  Fr={config.froude}")
        print(f" Hard BC: {config.hard_bc}  (alpha={config.bc_sharpness})")
        print(f"{'='*60}")

        # 1. Build SDF
        self.sdf_gen = SDFGenerator(config)
        if stl_path and os.path.exists(stl_path):
            self.sdf = self.sdf_gen.compute_sdf_from_stl(stl_path)
        else:
            if stl_path:
                print(f"[!] STL not found: {stl_path} — using analytical hull")
            self.sdf = self.sdf_gen.generate_analytical_hull_sdf()

        # 2. Build FNO input [B=1, 6, D, H, W]
        self.fno_input = self.sdf_gen.build_fno_input(
            self.sdf, config.reynolds, config.froude
        ).to(self.device)
        self.sdf_device = self.sdf.to(self.device)

        print(f"[OK] FNO input: {list(self.fno_input.shape)}")

        # 3. Build network
        self.model = FNO3d_NS_Solver(config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[OK] FNO3d_NS_Solver: {n_params:,} parameters")

        # 4. Build loss
        self.criterion = NavierStokesPINNLoss(
            config,
            dx=self.sdf_gen.dx,
            dy=self.sdf_gen.dy,
            dz=self.sdf_gen.dz,
        )

        # 5. Optimizer & scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6)

        # AMP scaler (CUDA only)
        self.scaler = None
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
            print("[OK] Mixed precision (AMP) enabled")

        self.history = []

    # ── Training Loop ────────────────────────────────────────────────────

    def train(self, epochs: Optional[int] = None):
        """Run the PINN training loop.

        Each iteration:
          1. Forward pass: FNO input → [u, v, w, p]
          2. Compute NS residuals via finite differences
          3. Backprop through spectral layers
        """
        n_epochs = epochs or self.config.epochs
        cfg = self.config

        # Loss weight scheduling: ramp up momentum weight gradually
        # (start with BCs to establish boundary structure first)
        def _mom_weight(epoch):
            # Linear ramp from 0.01 to 1.0 over first 20% of training
            ramp = min(1.0, epoch / (0.2 * n_epochs + 1))
            return 0.01 + 0.99 * ramp

        print(f"\n[>] Starting PINN training: {n_epochs} epochs")
        t0 = time.time()
        best_loss = float('inf')

        for epoch in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            # Adjust momentum weight
            cfg.lambda_momentum = _mom_weight(epoch)

            if self.scaler:  # AMP path
                with torch.amp.autocast('cuda'):
                    pred = self.model(self.fno_input, self.sdf_device)
                    losses = self.criterion(pred, self.sdf_device)
                self.scaler.scale(losses['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:  # Standard path
                pred = self.model(self.fno_input, self.sdf_device)
                losses = self.criterion(pred, self.sdf_device)
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()
            total = losses['total'].item()
            self.history.append(total)

            if total < best_loss:
                best_loss = total

            # Logging
            if epoch % 50 == 0 or epoch == n_epochs - 1:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:4d}/{n_epochs} | "
                    f"Total: {total:.2e} | "
                    f"Cont: {losses['continuity']:.2e} | "
                    f"Mom: {losses['momentum']:.2e} | "
                    f"NS: {losses['noslip']:.2e} | "
                    f"In: {losses['inlet']:.2e} | "
                    f"Out: {losses['outlet']:.2e} | "
                    f"LR: {lr:.1e} | {elapsed:.0f}s"
                )

        elapsed = time.time() - t0
        print(f"\n[OK] Training complete in {elapsed:.1f}s")
        print(f"    Best loss: {best_loss:.2e}")
        return self.history

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self) -> dict:
        """Run inference, return numpy arrays for visualisation."""
        self.model.eval()
        pred = self.model(self.fno_input, self.sdf_device)
        pred_np = pred.cpu().numpy()[0]  # [4, D, H, W]

        return {
            'u': pred_np[0], 'v': pred_np[1],
            'w': pred_np[2], 'p': pred_np[3],
            'sdf': self.sdf.numpy()[0, 0],
            'grid_x': self.sdf_gen.grid_x.numpy(),
            'grid_y': self.sdf_gen.grid_y.numpy(),
            'grid_z': self.sdf_gen.grid_z.numpy(),
        }

    # ── Save / Load ──────────────────────────────────────────────────────

    def save(self, path: str = None):
        path = path or os.path.join(
            os.path.dirname(__file__), '..', 'models', 'fno3d_pinn.pt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': vars(self.config),
            'history': self.history,
        }, path)
        print(f"[S] Saved: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt.get('history', [])
        print(f"[L] Loaded: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Quick Validation (shape + forward/backward)
# ═══════════════════════════════════════════════════════════════════════════════

def run_shape_test(config: SolverConfig):
    """Verify tensor dimensions through the full pipeline."""
    print("\n-- Shape Validation Test --")
    D, H, W = config.grid_dims

    gen = SDFGenerator(config)
    sdf = gen.generate_analytical_hull_sdf()
    assert sdf.shape == (1, 1, D, H, W), f"SDF shape: {sdf.shape}"

    fno_in = gen.build_fno_input(sdf)
    assert fno_in.shape == (1, 6, D, H, W), f"Input shape: {fno_in.shape}"

    model = FNO3d_NS_Solver(config)
    out = model(fno_in, sdf)
    assert out.shape == (1, 4, D, H, W), f"Output shape: {out.shape}"

    # Backward pass
    loss_fn = NavierStokesPINNLoss(config, gen.dx, gen.dy, gen.dz)
    losses = loss_fn(out, sdf)
    losses['total'].backward()

    grad_norm = sum(
        p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  SDF:     {list(sdf.shape)} OK")
    print(f"  Input:   {list(fno_in.shape)} OK")
    print(f"  Output:  {list(out.shape)} OK")
    print(f"  Loss:    {losses['total'].item():.4e} OK")
    print(f"  Grad norm:  {grad_norm:.4f} OK")
    print("-- All shape tests passed --\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SDF-Conditioned 3D-FNO Navier-Stokes PINN Solver")
    parser.add_argument('--stl', type=str, default=None,
                        help='Path to hull STL file')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu', action='store_true',
                        help='Force GPU if available')
    parser.add_argument('--hires', action='store_true',
                        help='Use high-res grid [64,32,128]')
    parser.add_argument('--test-only', action='store_true',
                        help='Run shape tests only')
    parser.add_argument('--hard-bc', type=bool, default=True)
    parser.add_argument('--re', type=float, default=1e6)
    parser.add_argument('--fr', type=float, default=0.26)
    args = parser.parse_args()

    # Build config
    config = SolverConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        reynolds=args.re,
        froude=args.fr,
        hard_bc=args.hard_bc,
    )

    if args.hires:
        config.grid_depth  = 64
        config.grid_height = 32
        config.grid_width  = 128
        config.modes_d = 16
        config.modes_h = 12
        config.modes_w = 32
        config.fno_width = 48

    if args.test_only:
        run_shape_test(config)
        return

    # Train
    trainer = PINNTrainer(config, stl_path=args.stl)
    trainer.train()
    trainer.save()

    # Quick inference check
    result = trainer.predict()
    u = result['u']
    print(f"\n[Result] u-velocity range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"[Result] u at inlet (x=0):   {u[:,:,0].mean():.4f}")
    print(f"[Result] u at outlet (x=-1): {u[:,:,-1].mean():.4f}")

    sdf = result['sdf']
    u_inside = u[sdf < 0]
    if len(u_inside) > 0:
        print(f"[Result] |u| inside hull: {np.abs(u_inside).max():.6f} "
              f"(should be ~0)")


if __name__ == '__main__':
    main()
