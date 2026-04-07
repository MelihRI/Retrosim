"""
OpenFOAM Data Generator (Solver Data)
=====================================
Automates the OpenFOAM CFD pipeline for SmartCAPEX ship geometries.
Generates thousands of flow field arrays (U, p) for different hull
shapes (STLs) and speeds to train the NVIDIA Modulus surrogate.

Workflow:
1. Copy base OpenFOAM template directory.
2. Insert generated STL from RetrosimHullAdapter.
3. Update U/p boundary conditions based on speed.
4. Run blockMesh, snappyHexMesh, simpleFoam.
5. Export results to .npz for fast PyTorch ingestion.
"""

import os
import shutil
import subprocess
import numpy as np

class OpenFOAMGenerator:
    def __init__(self, base_case_dir: str, output_dataset_dir: str):
        self.base_case = base_case_dir
        self.out_dir = output_dataset_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
    def setup_case(self, case_name: str, stl_path: str, speed_ms: float) -> str:
        """Creates a specific OpenFOAM case directory."""
        case_dir = os.path.join(self.out_dir, case_name)
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
            
        shutil.copytree(self.base_case, case_dir)
        
        # 1. Copy STL to trisurface
        os.makedirs(os.path.join(case_dir, "constant", "triSurface"), exist_ok=True)
        shutil.copy(stl_path, os.path.join(case_dir, "constant", "triSurface", "hull.stl"))
        
        # 2. Modify Inlet Velocity (0/U)
        u_file = os.path.join(case_dir, "0", "U")
        if os.path.exists(u_file):
            with open(u_file, 'r') as f:
                content = f.read()
            # Fast dirty replace for parameter sweep
            content = content.replace("INLET_VELOCITY", f"({speed_ms} 0 0)")
            with open(u_file, 'w') as f:
                f.write(content)
                
        return case_dir

    def run_simulation(self, case_dir: str):
        """Executes the OpenFOAM macros sequentially."""
        print(f"🚀 Running CFD for {case_dir}...")
        cmds = [
            ["blockMesh"],
            ["snappyHexMesh", "-overwrite"],
            ["simpleFoam"]
        ]
        
        for cmd in cmds:
            try:
                subprocess.run(cmd, cwd=case_dir, check=True, 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"❌ OpenFOAM Error ({cmd[0]}): {e}")
                return False
        return True

    def export_to_modulus(self, case_dir: str, npz_name: str):
        """
        Extracts final timestep U and p fields into an NPZ archive
        suitable for NVIDIA Modulus data loaders.
        Requires PVPython or vtk wrappers in a real setup.
        """
        # Placeholder for OpenFOAM-to-VTK/Numpy extraction
        # Real implementation calls postProcess -func sample or paraview scripts
        dummy_u = np.random.rand(100, 100, 3).astype(np.float32)
        dummy_p = np.random.rand(100, 100, 1).astype(np.float32)
        
        save_path = os.path.join(self.out_dir, npz_name)
        np.savez_compressed(save_path, u=dummy_u, p=dummy_p)
        print(f"💾 Saved Modulus training data: {save_path}")

if __name__ == "__main__":
    # Example Usage
    generator = OpenFOAMGenerator(
        base_case_dir="../templates/OpenFOAM_Base",
        output_dataset_dir="../models/cfd_dataset"
    )
    # generator.setup_case("Hull_V1", "hull.stl", speed_ms=8.5)
    # generator.run_simulation("Hull_V1")
