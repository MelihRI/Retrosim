import os
import tempfile
import numpy as np
import pyjet

class PyJetFlowField:
    """
    Real-time Fluid Engine adapter using PyJet FLIP Solver.
    """
    def __init__(self, verts, faces, ship_L=1.5, ship_B=0.3, ship_T=0.15, ship_speed=12.0):
        self.ship_L = ship_L
        self.ship_B = ship_B
        self.ship_T = ship_T
        self.ship_speed = ship_speed
        
        # Real-time lower resolution
        self.resX = 48
        self.resY = 24
        self.resZ = 16
        
        # Slower time step for real-time play
        self.dt = 1.0 / 60.0
        
        self._init_solver(verts, faces)
        
    def _init_solver(self, verts, faces):
        print(f"🌊 Initializing PyJet FLIP Solver ({self.resX}x{self.resY}x{self.resZ})")
        # 1. Create solver
        self.solver = pyjet.FlipSolver3(resolutionX=self.resX, resolutionY=self.resY, resolutionZ=self.resZ)
        self.solver.useSpikyKernel = True
        
        # 2. Hull Mesh
        mesh = pyjet.TriangleMesh3()
        
        # STL to OBJ temporary conversion
        with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for t in faces:
                f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
            temp_path = f.name
            
        mesh.readObj(temp_path)
        os.remove(temp_path)
        
        # Calculate domain mappings based on L
        # domain stretches from -1.5*L to 3*L length
        bound_x = self.ship_L * 4.5
        domain_scale_x = self.resX / bound_x
        
        # Convert mesh to local grid coordinates
        # Pyjet domains are [0, resX], so we translate the mesh.
        # Actually, let's keep Pyjet's domain as a BoundingBox inside the world coords.
        # pyjet's GridSystemData has bounding box, wait... default bounding box is (0,0,0) to sizes.
        # We can just tell Pyjet solver physical domain size.
        grid_system = self.solver.gridSystemData
        # grid space: dx = bound_x/resX
        grid_spacing = bound_x / self.resX
        grid_system.origin = (-1.5 * self.ship_L, -2.0 * self.ship_B, -2.5 * self.ship_T)
        grid_system.gridSpacing = (grid_spacing, grid_spacing, grid_spacing)
        
        # Create Implicit Surface
        surface = pyjet.ImplicitSurface3(mesh)
        collider = pyjet.RigidBodyCollider3(surface)
        self.solver.collider = collider
        
        # 3. Water Volume
        # Represented as a bounding box representing water
        water_bbox = pyjet.BoundingBox3D(
            (-float("inf"), -float("inf"), -float("inf")),
            (float("inf"), float("inf"), -self.ship_T * 0.1)  # Waterline is around z = 0, draft is below
        )
        water_surface = pyjet.ImplicitSurface3(water_bbox)
        
        # Emitter to create particles
        emitter = pyjet.VolumeParticleEmitter3(
            implicitSurface=water_surface,
            maxRegion=pyjet.BoundingBox3D(
                grid_system.origin,
                (grid_system.origin[0] + grid_spacing*self.resX,
                 grid_system.origin[1] + grid_spacing*self.resY,
                 grid_system.origin[2] + grid_spacing*self.resZ)
            ),
            spacing=grid_spacing / 2.0
        )
        self.solver.particleEmitter = emitter
        
    def step(self):
        """Advances simulation by one tick."""
        frame = pyjet.Frame(0, self.dt)
        self.solver.update(frame)
        
    def get_particles(self):
        """Returns numpy array of particle positions."""
        particles = np.array(self.solver.particleSystemData.positions, copy=False)
        return particles
