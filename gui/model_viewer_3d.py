import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QTimer
import random
import os

try:
    from pxr import Usd, UsdGeom
    HAS_USD = True
except ImportError:
    HAS_USD = False

# --- MATH UTILS ---
def translation(displacement):
    t = np.identity(4, dtype=np.float32)
    t[0, 3] = displacement[0]
    t[1, 3] = displacement[1]
    t[2, 3] = displacement[2]
    return t

def scaling(scale):
    s = np.identity(4, dtype=np.float32)
    s[0, 0] = scale[0]
    s[1, 1] = scale[1]
    s[2, 2] = scale[2]
    s[3, 3] = 1
    return s

class AABB:
    def __init__(self, mins, maxs):
        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)
    
    def scale(self, s):
        self.mins *= s
        self.maxs *= s

    def ray_hit(self, start, direction, mat):
        # Very simplified ray-AABB intersection for performance
        # In a real app, this would be more robust.
        # Transforming ray to model space
        inv_mat = np.linalg.inv(mat)
        start_m = (inv_mat @ np.append(start, 1.0))[:3]
        dir_m = (inv_mat @ np.append(direction, 0.0))[:3]
        
        # Simple slab method approximation
        tmin = (self.mins - start_m) / (dir_m + 1e-6)
        tmax = (self.maxs - start_m) / (dir_m + 1e-6)
        
        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)
        
        tnear = np.max(t1)
        tfar = np.min(t2)
        
        if tnear <= tfar and tfar > 0:
            return True, tnear
        return False, 0

# --- NODE HIERARCHY ---
class Node:
    def __init__(self):
        self.color = [random.random(), random.random(), random.random()]
        self.aabb = AABB([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
        self.translation_matrix = np.identity(4, dtype=np.float32)
        self.scaling_matrix = np.identity(4, dtype=np.float32)
        self.selected = False
        self.depth = 0
        self.lod_level = 0 # 0: Full, 1: Half, 2: Low

    def set_lod(self, level):
        self.lod_level = level

    def swap_axes(self):
        pass

    def optimize_geometry(self):
        pass

    def render(self):
        glPushMatrix()
        # Note: OpenGL expects column-major, but numpy is row-major. Transpose needed.
        glMultMatrixf(self.translation_matrix.T)
        glMultMatrixf(self.scaling_matrix.T)
        
        if self.selected:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.3, 0.3, 0.3, 1.0])
            glColor3f(1.0, 1.0, 0.0) # Highlight yellow
        else:
            glMaterialfv(GL_FRONT, GL_EMISSION, [0.0, 0.0, 0.0, 1.0])
            glColor3fv(self.color)
            
        self.render_self()
        glPopMatrix()

    def render_self(self):
        raise NotImplementedError()

    def translate(self, x, y, z):
        self.translation_matrix = self.translation_matrix @ translation([x, y, z])

    def scale(self, s):
        self.scaling_matrix = self.scaling_matrix @ scaling([s, s, s])
        self.aabb.scale(s)

    def pick(self, start, direction, mat):
        newmat = mat @ self.translation_matrix @ np.linalg.inv(self.scaling_matrix)
        return self.aabb.ray_hit(start, direction, newmat)

class Cube(Node):
    def render_self(self):
        # Draw a simple unit cube
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(-0.5, -0.5, 0.5); glVertex3f(0.5, -0.5, 0.5)
        glVertex3f(0.5, 0.5, 0.5); glVertex3f(-0.5, 0.5, 0.5)
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(-0.5, -0.5, -0.5); glVertex3f(-0.5, 0.5, -0.5)
        glVertex3f(0.5, 0.5, -0.5); glVertex3f(0.5, -0.5, -0.5)
        # ... others simplified for code size
        glEnd()

class Sphere(Node):
    def render_self(self):
        quad = gluNewQuadric()
        gluSphere(quad, 0.5, 16, 16)

class HierarchicalNode(Node):
    def __init__(self):
        super().__init__()
        self.child_nodes = []
    
    def render_self(self):
        for child in self.child_nodes:
            child.render()

class Particle:
    def __init__(self, bounds):
        self.bounds = bounds
        self.reset()
        self.life = random.random()
        
    def reset(self):
        # Start at random positions within the front part of the domain (inlet)
        self.pos = np.array([
            random.uniform(self.bounds.mins[0], self.bounds.mins[0] + 0.5),
            random.uniform(self.bounds.mins[1], self.bounds.maxs[1]),
            random.uniform(self.bounds.mins[2], self.bounds.maxs[2])
        ], dtype=np.float32)
        self.life = 1.0
        self.velocity = np.array([0.1, 0, 0], dtype=np.float32)

class CFDNode(Node):
    def __init__(self, results):
        super().__init__()
        self.results = results
        self.aabb = AABB([-1, -1, -0.5], [2, 1, 0.5])
        self.particles = [Particle(self.aabb) for _ in range(300)]
        self.time_step = 0.016 # ~60fps
        
    def render_self(self):
        """Advanced 'Omniverse' style visualization"""
        # 1. Particles update
        self.update_particles(self.time_step)
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Glow effect
        
        # Draw Particles
        glPointSize(3.0)
        glBegin(GL_POINTS)
        for p in self.particles:
            # Color based on velocity mag or life
            glColor4f(0.2, 0.8, 1.0, p.life * 0.7)
            glVertex3fv(p.pos)
        glEnd()

        # 2. Pressure Hologram
        X, Y = self.results['X'], self.results['Y']
        P = self.results['P']
        rows, cols = X.shape
        step = 2
        for z_off in [-0.2, 0.2]:
            glBegin(GL_QUADS)
            for i in range(0, rows - 1, step):
                for j in range(0, cols - 1, step):
                    for di, dj in [(0,0), (step,0), (step,step), (0,step)]:
                        ii, jj = min(i+di, rows-1), min(j+dj, cols-1)
                        norm_p = np.clip((P[ii, jj] + 1.2) / 2.4, 0, 1)
                        glColor4f(0.0, 0.5, norm_p, 0.1) 
                        glVertex3f(X[ii, jj], Y[ii, jj], z_off)
            glEnd()
            
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

    def update_particles(self, dt):
        X, Y = self.results['X'], self.results['Y']
        U, V = self.results['U'], self.results['V']
        
        for p in self.particles:
            p.life -= dt * 0.2
            if p.life <= 0 or p.pos[0] > self.aabb.maxs[0]:
                p.reset()
            
            # Simple bilinear map to grid
            try:
                ix = int(((p.pos[0] - (-0.5)) / 2.3) * (X.shape[1]-1))
                iy = int(((p.pos[1] - (-0.8)) / 1.6) * (X.shape[0]-1))
                ix = np.clip(ix, 0, X.shape[1]-1); iy = np.clip(iy, 0, X.shape[0]-1)
                p.velocity[0] = U[iy, ix] * 0.5
                p.velocity[1] = V[iy, ix] * 0.5
            except: pass
            
            p.pos += p.velocity * dt * 8.0

class ShipHull(Node):
    def __init__(self, loa=190.0, lbp=182.0, beam=32.2, draft=12.5, depth=18.0, cb=0.82, cp=0.84, cm=0.98,
                 bow_height=4.0, stern_height=2.0, bulb_l=5.5, bulb_r=2.2, stern_s=0.8,
                 prop_d=6.5, prop_b=4, rudder_h=7.0,
                 vessel_type="Bulk Carrier", retrofit_components=None):
        super().__init__()
        self.loa = loa
        self.lbp = lbp
        self.beam = beam
        self.draft = draft
        self.depth = depth
        self.cb = cb
        self.cp = cp
        self.cm = cm
        self.bow_h = bow_height
        self.stern_h = stern_height
        self.bulb_l = bulb_l
        self.bulb_r = bulb_r
        self.stern_s = stern_s
        self.prop_dia = prop_d
        self.prop_blades = prop_b
        self.rudder_h = rudder_h
        
        self.vessel_type = vessel_type
        self.retrofit_components = retrofit_components or []
        
        self.l_scale = (loa / 190.0) * 8.0
        self.lbp_scale = (lbp / 190.0) * 8.0
        self.b_scale = (beam / 32.2) * 2.5
        self.t_scale = (draft / 12.5) * 1.5
        self.d_scale = (depth / 12.5) * 1.5
        
        self.generate_hull_mesh()

    def generate_hull_mesh(self):
        """
        Generates a realistic hull mesh using naval architecture principles.
        
        Uses:
        - Sectional Area Curve (SAC) for longitudinal volume distribution
        - Lewis form sections for cross-section shapes
        - Proper bow and stern fairing
        - Sheer curve and deck camber
        - Bulbous bow integration
        """
        self.stations = 60  # Higher resolution for smooth curves
        self.pts_per_station = 32  # More points for smooth sections
        
        L = self.l_scale
        B = self.b_scale / 2.0  # Half-beam
        T = self.t_scale
        D = self.d_scale
        
        # Form coefficients
        Cb = self.cb
        Cp = self.cp
        Cm = self.cm
        
        # Derived coefficients
        # Cwp (Waterplane coefficient) - estimated from Cb
        Cwp = 0.18 + 0.86 * Cb
        
        # Lcb position (fraction from midship, positive = forward)
        # For bulk carriers, Lcb is usually slightly aft
        Lcb = -0.02 + 0.1 * (Cb - 0.7)
        
        # Rise of floor (deadrise angle for bilge)
        bilge_radius = B * np.sqrt(2.0 * (1.0 - Cm) / np.pi) if Cm < 0.99 else B * 0.05
        bilge_radius = np.clip(bilge_radius, B * 0.05, B * 0.4)
        
        mesh_pts = []
        
        for i in range(self.stations + 1):
            # Normalized position: 0 = stern, 1 = bow
            xi = i / self.stations
            x = (xi - 0.5) * L
            
            # ========================================
            # 1. SECTIONAL AREA CURVE (SAC)
            # ========================================
            # Using a more realistic SAC based on Cp
            # Lackenby transformation for realistic longitudinal distribution
            
            if xi < 0.5:
                # Aft body (stern to midship)
                t = xi * 2.0  # 0 to 1
                # Stern is sharper, use power function
                n_aft = 1.5 + (1.0 - Cp) * 3.0
                sac = np.power(t, 1.0/n_aft)
            else:
                # Fore body (midship to bow)
                t = (xi - 0.5) * 2.0  # 0 to 1
                # Forward end more gradual for wave-piercing
                n_fwd = 2.0 + (1.0 - Cp) * 2.5
                sac = 1.0 - np.power(t, n_fwd)
            
            # Adjust for Cb - fuller ships have more parallel midbody
            if 0.25 < xi < 0.75:
                parallel_factor = (Cb - 0.5) / 0.4
                sac = sac + (1.0 - sac) * parallel_factor * 0.3
            
            sac = np.clip(sac, 0.01, 1.0)
            
            # ========================================
            # 2. WATERPLANE SHAPE
            # ========================================
            # Waterplane is narrower than SAC suggests at ends
            
            if xi < 0.15:  # Stern region
                wp_factor = np.power(xi / 0.15, 0.7)
                # Transom stern - maintain some width
                if self.stern_s > 0.5:
                    wp_factor = max(wp_factor, 0.3 * self.stern_s)
            elif xi > 0.85:  # Bow region  
                wp_factor = np.power((1.0 - xi) / 0.15, 0.5)
            else:
                wp_factor = 1.0
            
            # Current beam at this station
            current_beam = B * np.sqrt(sac) * wp_factor
            current_beam = max(current_beam, 0.02 * B)  # Minimum for numerical stability
            
            # ========================================
            # 3. SHEER CURVE
            # ========================================
            # Classic sheer curve - higher at bow and stern
            
            x_rel = (xi - 0.5) * 2.0  # -1 to 1
            
            # Bow sheer (more pronounced)
            if x_rel > 0:
                sheer = self.bow_h * 0.15 * np.power(x_rel, 2.0)
            else:
                # Stern sheer (gentler)
                sheer = self.stern_h * 0.10 * np.power(-x_rel, 1.5)
            
            deck_z = D + sheer
            
            # ========================================
            # 4. SECTION SHAPE (Lewis Form approximation)
            # ========================================
            # Generate points for half-section (starboard side)
            
            station_pts = []
            
            # Section shape parameters vary along length
            if xi < 0.2:
                # Stern sections: U-shaped for propeller clearance
                section_fullness = 0.5 + 0.4 * (xi / 0.2)
                deadrise = 0.15 * (1.0 - xi / 0.2)  # More deadrise at stern
            elif xi > 0.8:
                # Bow sections: V-shaped for wave cutting
                bow_factor = (xi - 0.8) / 0.2
                section_fullness = 0.9 - 0.5 * bow_factor
                deadrise = 0.2 * bow_factor
            else:
                # Parallel midbody: box-like with bilge
                section_fullness = 0.85 + 0.1 * Cm
                deadrise = 0.02
            
            for j in range(self.pts_per_station):
                # Parametric curve from centerline-bottom to deck edge
                s = j / (self.pts_per_station - 1)  # 0 to 1
                
                # Split section into regions
                if s < 0.15:
                    # KEEL/BOTTOM region (flat with deadrise)
                    t = s / 0.15
                    y = current_beam * t * section_fullness * 0.7
                    z = -T + deadrise * T * (1.0 - np.cos(t * np.pi / 2))
                    
                elif s < 0.35:
                    # BILGE region (curved transition)
                    t = (s - 0.15) / 0.20
                    angle = t * np.pi / 2.0
                    
                    # Start from where bottom ended
                    y_start = current_beam * section_fullness * 0.7
                    z_start = -T + deadrise * T
                    
                    # Bilge circle
                    r = bilge_radius * np.sqrt(sac)
                    y = y_start + r * np.sin(angle)
                    z = z_start + r * (1.0 - np.cos(angle))
                    
                elif s < 0.85:
                    # SIDE region (nearly vertical with flare)
                    t = (s - 0.35) / 0.50
                    
                    # Vertical side position
                    y_bilge_end = current_beam * section_fullness * 0.7 + bilge_radius * np.sqrt(sac)
                    z_bilge_end = -T + deadrise * T + bilge_radius * np.sqrt(sac)
                    
                    # Flare factor - increases towards bow
                    flare = 0.0
                    if xi > 0.7:
                        flare = 0.15 * ((xi - 0.7) / 0.3) * t
                    
                    # Tumblehome for stern
                    tumblehome = 0.0
                    if xi < 0.2:
                        tumblehome = -0.05 * ((0.2 - xi) / 0.2) * t
                    
                    y = y_bilge_end + (current_beam - y_bilge_end + current_beam * flare + current_beam * tumblehome) * t
                    z = z_bilge_end + (deck_z - z_bilge_end) * t
                    
                else:
                    # DECK EDGE region (round over or knuckle)
                    t = (s - 0.85) / 0.15
                    
                    # Deck camber (slight curve)
                    camber = 0.02 * B * (1.0 - t)
                    
                    y = current_beam + current_beam * 0.05 * (xi > 0.7) * ((xi - 0.7) / 0.3)
                    z = deck_z - camber
                
                # ========================================
                # 5. BOW REFINEMENT
                # ========================================
                if xi > 0.92:
                    # Sharp bow convergence
                    bow_taper = (xi - 0.92) / 0.08
                    y *= (1.0 - bow_taper * 0.95)
                    
                    # Bow rake (forward lean)
                    if z > 0:
                        x_offset = bow_taper * 0.3 * L * 0.1
                    else:
                        x_offset = 0
                    x = (xi - 0.5) * L + x_offset
                
                # ========================================
                # 6. STERN REFINEMENT
                # ========================================
                if xi < 0.08:
                    # Stern convergence (but maintain transom shape)
                    stern_taper = 1.0 - (xi / 0.08)
                    
                    if self.stern_s > 0.5:  # Transom stern
                        # Keep some width for transom
                        y *= (1.0 - stern_taper * 0.5)
                    else:  # Cruiser stern
                        y *= (1.0 - stern_taper * 0.85)
                    
                    # Skeg area - deep and narrow
                    if s < 0.3:
                        y *= (1.0 - stern_taper * 0.6)
                
                station_pts.append((x, max(y, 0.001), z))
            
            mesh_pts.append(station_pts)
        
        self.hull_pts = mesh_pts
        self.calculate_normals()

    def calculate_normals(self):
        """Calculates smooth normals for the hull mesh"""
        self.normals = []
        for i in range(len(self.hull_pts)):
            station_normals = []
            for j in range(len(self.hull_pts[i])):
                # Cross product of adjacent vectors
                next_i = min(i + 1, len(self.hull_pts) - 1)
                prev_i = max(i - 1, 0)
                next_j = min(j + 1, len(self.hull_pts[i]) - 1)
                prev_j = max(j - 1, 0)
                
                v1 = np.array(self.hull_pts[next_i][j]) - np.array(self.hull_pts[prev_i][j])
                v2 = np.array(self.hull_pts[i][next_j]) - np.array(self.hull_pts[i][prev_j])
                
                n = np.cross(v2, v1)
                norm = np.linalg.norm(n)
                if norm > 0: n /= norm
                station_normals.append(n)
            self.normals.append(station_normals)

    def render_self(self):
        """Main rendering pipeline for the vessel"""
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, [50.0])
        
        # 1. HULL SURFACE
        for i in range(len(self.hull_pts) - 1):
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(self.pts_per_station):
                p1 = self.hull_pts[i][j]
                p2 = self.hull_pts[i+1][j]
                n1 = self.normals[i][j]
                n2 = self.normals[i+1][j]
                
                if p1[2] < 0: glColor3f(0.06, 0.08, 0.15)
                else: glColor3f(0.12, 0.16, 0.22)
                
                glNormal3fv(n1); glVertex3fv(p1)
                glNormal3fv(n2); glVertex3fv(p2)
            glEnd()
            
            # Starboard Mirror
            glBegin(GL_TRIANGLE_STRIP)
            for j in range(self.pts_per_station):
                p1, p2 = self.hull_pts[i][j], self.hull_pts[i+1][j]
                n1, n2 = self.normals[i][j], self.normals[i+1][j]
                glNormal3f(n1[0], -n1[1], n1[2]); glVertex3f(p1[0], -p1[1], p1[2])
                glNormal3f(n2[0], -n2[1], n2[2]); glVertex3f(p2[0], -p2[1], p2[2])
            glEnd()

        # 2. TRANSOM STERN & DECK
        glBegin(GL_QUAD_STRIP); glColor3f(0.04, 0.05, 0.06); glNormal3f(-1, 0, 0)
        for j in range(self.pts_per_station):
            p = self.hull_pts[0][j]; glVertex3f(p[0], p[1], p[2]); glVertex3f(p[0], -p[1], p[2])
        glEnd()

        glColor3f(0.2, 0.22, 0.25); glBegin(GL_QUAD_STRIP); glNormal3f(0, 0, 1)
        for i in range(len(self.hull_pts)):
            p = self.hull_pts[i][-1]; glVertex3f(p[0], p[1], p[2]); glVertex3f(p[0], -p[1], p[2])
        glEnd()

    def render_special_features(self):
        L = self.l_scale; B = self.b_scale / 2.0; T = self.t_scale
        
        # Helper for Deck Height
        def get_deck_z(x_pos):
            idx = int(np.clip((x_pos/L + 0.5) * self.stations, 0, self.stations))
            return self.hull_pts[idx][-1][2]

        if self.vessel_type in ["Bulk Carrier", "General Cargo"]:
            glColor3f(0.15, 0.15, 0.18)
            for x_p in np.linspace(-L*0.1, L*0.35, 4):
                glPushMatrix()
                glTranslatef(x_p, 0, get_deck_z(x_p) + 0.05)
                self.draw_box(L*0.1, B*1.4, 0.15)
                glPopMatrix()
                
        elif self.vessel_type == "Container":
            for x_p in np.linspace(-L*0.15, L*0.35, 6):
                glColor3f(random.uniform(0.2,0.6), random.uniform(0.1,0.3), random.uniform(0.2,0.7))
                glPushMatrix()
                glTranslatef(x_p, 0, get_deck_z(x_p) + 0.8)
                self.draw_box(L*0.08, B*1.6, T*1.2)
                glPopMatrix()

        elif self.vessel_type == "Yelkenli":
            glDisable(GL_CULL_FACE)
            for x_p in [-L*0.1, L*0.2]:
                z_base = get_deck_z(x_p)
                glColor3f(0.4, 0.25, 0.1); glPushMatrix(); glTranslatef(x_p, 0, z_base + T*2.5); self.draw_box(0.15, 0.15, T*5.0)
                glColor4f(0.95, 0.95, 1.0, 0.8); glBegin(GL_TRIANGLES); glVertex3f(0,0,T*2); glVertex3f(L*0.2,0,-T*2); glVertex3f(0,0,-T*2); glEnd(); glPopMatrix()
            glEnable(GL_CULL_FACE)

        elif self.vessel_type == "Fastbot":
            glColor3f(0.1, 0.1, 0.1)
            for y_p in [-B*0.4, B*0.4]:
                glPushMatrix(); glTranslatef(-L*0.48, y_p, -T*0.3); self.draw_box(0.5, 0.5, 1.0); glPopMatrix()

        # Retrofit: Flettner
        if "flettner_rotor" in self.retrofit_components:
            glColor3f(0.95, 0.95, 0.95)
            for x_p in [-L*0.05, L*0.15, L*0.3]:
                if self.vessel_type == "Container" and x_p > 0: continue
                z_base = get_deck_z(x_p)
                glPushMatrix(); glTranslatef(x_p, 0, z_base + T*0.5)
                glColor3f(0.2, 0.2, 0.25); self.draw_box(0.6, 0.6, 0.5) # Base
                glColor3f(1.0, 1.0, 1.0); glTranslatef(0, 0, T*1.2); self.draw_cylinder(0.35 * self.b_scale, T * 2.8); glPopMatrix()

    def render_superstructure(self):
        # Position at approx 15% from stern
        x_pos = -L * 0.35
        idx = int(np.clip((x_pos/self.l_scale + 0.5) * self.stations, 0, self.stations))
        deck_z = self.hull_pts[idx][-1][2]
        
        glPushMatrix(); glTranslatef(x_pos, 0, deck_z)
        glColor3f(0.9, 0.92, 0.95); self.draw_box(2.4, self.b_scale * 0.75, 1.6) # Main Castle
        # Bridge
        glTranslatef(-0.2, 0, 1.6); glColor3f(0.8, 0.82, 0.85); self.draw_box(1.8, self.b_scale * 0.85, 1.0)
        # Funnel
        glTranslatef(-1.2, 0, 0.2); glColor3f(0.15, 0.15, 0.18); self.draw_box(0.8, self.b_scale * 0.25, 2.0)
        glPopMatrix()

    def render_propulsion(self):
        L = self.l_scale; T = self.t_scale
        p_dia = (self.prop_dia / 32.2) * 2.5
        r_h = (self.rudder_h / 12.5) * 1.5
        
        # RUDDER (At the very end of the narrowed hull)
        glColor3f(0.1, 0.1, 0.15)
        glPushMatrix()
        glTranslatef(-L * 0.49, 0, -T * 0.75)
        self.draw_box(0.35, 0.08, r_h)
        glPopMatrix()
        
        # PROPELLER (Positioned in the 'tapered' area)
        glPushMatrix()
        glTranslatef(-L * 0.45, 0, -T * 0.9)
        # Hub
        glColor3f(0.2, 0.2, 0.2); quad = gluNewQuadric(); gluSphere(quad, p_dia*0.18, 20, 20)
        # Blades
        glColor3f(0.85, 0.65, 0.2) # Premium Bronze
        for i in range(self.prop_blades):
            glPushMatrix()
            glRotatef(i * (360.0 / self.prop_blades), 1, 0, 0)
            glRotatef(25, 0, 1, 0) # Pitch
            glTranslatef(0, p_dia * 0.3, 0)
            self.draw_box(0.06, p_dia * 0.5, p_dia * 0.2)
            glPopMatrix()
        glPopMatrix()

    def render_bulb(self):
        L = self.l_scale; T = self.t_scale
        b_l = (self.bulb_l / 190.0) * 8.0; b_r = (self.bulb_r / 32.2) * 2.5
        glPushMatrix()
        # Position bulb slightly lower and forward
        glTranslatef(L * 0.48, 0, -T * 0.7)
        glScalef(2.0, 0.9, 0.9) # More elliptical
        glColor3f(0.05, 0.08, 0.12)
        quad = gluNewQuadric(); gluSphere(quad, b_r, 32, 32)
        glPopMatrix()

    def render_superstructure(self):
        stern_station = self.hull_pts[int(self.stations * 0.15)]
        deck_z = stern_station[-1][2]; stern_x = stern_station[-1][0]
        glPushMatrix(); glTranslatef(stern_x + 0.6, 0, deck_z)
        glColor3f(0.85, 0.85, 0.88); self.draw_box(2.2, self.b_scale * 0.7, 1.4) # Accom
        glTranslatef(-0.2, 0, 1.4); glColor3f(0.75, 0.78, 0.8); self.draw_box(1.6, self.b_scale * 0.8, 0.9) # Bridge
        glTranslatef(-1.0, 0, 0.2); glColor3f(0.1, 0.1, 0.15); self.draw_box(0.7, self.b_scale * 0.2, 1.8) # Funnel
        glPopMatrix()

    def render_outlines(self):
        glDisable(GL_LIGHTING); glLineWidth(2.5); glColor4f(0.0, 1.0, 1.0, 1.0)
        glBegin(GL_LINES)
        for i in range(len(self.hull_pts)-1):
            p1 = self.hull_pts[i][self.pts_per_station // 2]; p2 = self.hull_pts[i+1][self.pts_per_station // 2]
            glVertex3f(p1[0], p1[1], 0); glVertex3f(p2[0], p2[1], 0)
            glVertex3f(p1[0], -p1[1], 0); glVertex3f(p2[0], -p2[1], 0)
        glEnd()
        glLineWidth(0.5); glColor3f(0.4, 0.4, 0.4); glBegin(GL_LINES)
        for i in range(0, len(self.hull_pts), 4):
            for j in range(len(self.hull_pts[i]) - 1):
                p1 = self.hull_pts[i][j]; p2 = self.hull_pts[i][j+1]
                glVertex3fv(p1); glVertex3fv(p2); glVertex3f(p1[0], -p1[1], p1[2]); glVertex3f(p2[0], -p2[1], p2[2])
        glEnd(); glEnable(GL_LIGHTING)

    def draw_box(self, l, b, h):
        glBegin(GL_QUADS)
        # Top
        glNormal3f(0, 0, 1)
        glVertex3f(l/2, b/2, h/2); glVertex3f(-l/2, b/2, h/2)
        glVertex3f(-l/2, -b/2, h/2); glVertex3f(l/2, -b/2, h/2)
        # Front
        glNormal3f(1, 0, 0)
        glVertex3f(l/2, b/2, h/2); glVertex3f(l/2, -b/2, h/2)
        glVertex3f(l/2, -b/2, -h/2); glVertex3f(l/2, b/2, -h/2)
        # Sides
        glNormal3f(0, 1, 0)
        glVertex3f(l/2, b/2, h/2); glVertex3f(l/2, b/2, -h/2)
        glVertex3f(-l/2, b/2, -h/2); glVertex3f(-l/2, b/2, h/2)
        glNormal3f(0, -1, 0)
        glVertex3f(l/2, -b/2, h/2); glVertex3f(-l/2, -b/2, h/2)
        glVertex3f(-l/2, -b/2, -h/2); glVertex3f(l/2, -b/2, -h/2)
        # Back
        glNormal3f(-1, 0, 0)
        glVertex3f(-l/2, b/2, h/2); glVertex3f(-l/2, b/2, -h/2)
        glVertex3f(-l/2, -b/2, -h/2); glVertex3f(-l/2, -b/2, h/2)
        glEnd()


class STLHull(Node):
    def __init__(self, file_path):
        super().__init__()
        self.mesh_data = mesh.Mesh.from_file(file_path)
        # Calculate bounds for AABB
        minx = maxx = miny = maxy = minz = maxz = 0
        if len(self.mesh_data.points) > 0:
            minx = np.min(self.mesh_data.x)
            maxx = np.max(self.mesh_data.x)
            miny = np.min(self.mesh_data.y)
            maxy = np.max(self.mesh_data.y)
            minz = np.min(self.mesh_data.z)
            maxz = np.max(self.mesh_data.z)
        self.aabb = AABB([minx, miny, minz], [maxx, maxy, maxz])
        
        # Center the model and scale it to a reasonable size
        self.center_and_scale()

    def center_and_scale(self):
        extent = self.aabb.maxs - self.aabb.mins
        max_dim = np.max(extent)
        scale = 5.0 / max_dim if max_dim > 0 else 1.0
        
        # Center XY at origin, Z bottom at origin (sits on floor)
        center_x = (self.aabb.maxs[0] + self.aabb.mins[0]) / 2.0
        center_y = (self.aabb.maxs[1] + self.aabb.mins[1]) / 2.0
        z_offset = self.aabb.mins[2]  # Move bottom to z=0
        
        # Apply transformation to the mesh points for performance (once)
        self.mesh_data.x -= center_x
        self.mesh_data.y -= center_y
        self.mesh_data.z -= z_offset  # Bottom at Z=0
        
        self.mesh_data.x *= scale
        self.mesh_data.y *= scale
        self.mesh_data.z *= scale
        
        # Re-calc AABB
        new_minz = 0.0
        new_maxz = (self.aabb.maxs[2] - self.aabb.mins[2]) * scale
        self.aabb = AABB([-2.5, -2.5, new_minz], [2.5, 2.5, new_maxz])
        
        print(f"STL Model loaded: bounds Z=[0, {new_maxz:.2f}]")
        
        # Prepare Flattened Arrays for optimized rendering
        self.prepare_render_arrays()

    def prepare_render_arrays(self):
        """Flatten mesh and compute smooth vertex normals for High-Fidelity shading"""
        v = self.mesh_data.vectors.reshape(-1, 3)
        n = self.mesh_data.normals
        n_repeated = np.repeat(n, 3, axis=0)

        # Smooth normals calculation
        # Dict merging vertices within a small threshold to average their adjacent face normals
        v_dict = {}
        for i, vertex in enumerate(v):
            v_k = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
            if v_k not in v_dict:
                v_dict[v_k] = np.zeros(3, dtype=np.float32)
            v_dict[v_k] += n_repeated[i]

        for k in v_dict:
            norm = np.linalg.norm(v_dict[k])
            if norm > 0: v_dict[k] /= norm

        smooth_normals = np.zeros_like(v, dtype=np.float32)
        for i, vertex in enumerate(v):
            v_k = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
            smooth_normals[i] = v_dict[v_k]

        self.points_vbo = v.astype(np.float32)
        self.normals_vbo = smooth_normals

    def render_self(self):
        # Premium glossy material
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, [128.0])
        # Deep metallic obsidian color
        glColor3f(0.12, 0.16, 0.22)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, self.points_vbo)
        glNormalPointer(GL_FLOAT, 0, self.normals_vbo)
        
        # Draw solid geometry (smoothly shaded)
        glDrawArrays(GL_TRIANGLES, 0, len(self.points_vbo))
        
        # Removed buggy wireframe overlay for clean industrial look
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

class USDHull(Node):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.vertices = []
        self.normals = []
        self.indices = []
        
        if HAS_USD:
            self.load_usd_data(file_path)
        else:
            print("Warning: pxr (usd-core) not installed. Cannot load USD.")

    def load_usd_data(self, file_path):
        stage = Usd.Stage.Open(file_path)
        all_pts_list = []
        all_indices_list = []
        
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_geom = UsdGeom.Mesh(prim)
                pts = mesh_geom.GetPointsAttr().Get()
                f_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
                f_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
                
                if pts and f_indices:
                    offset = sum(len(p) for p in all_pts_list)
                    pts_np = np.array(pts, dtype=np.float32)
                    all_pts_list.append(pts_np)
                    
                    indices_np = np.array(f_indices, dtype=np.int32)
                    counts_np = np.array(f_counts, dtype=np.int32)
                    
                    # --- OPTIMIZED TRIANGULATION ---
                    if np.all(counts_np == 3):
                        # Pure triangles: reshape directly
                        tris = indices_np.reshape(-1, 3) + offset
                        all_indices_list.append(tris)
                    elif np.all(counts_np == 4):
                        # Pure quads: split into triangles
                        quads = indices_np.reshape(-1, 4) + offset
                        tris1 = quads[:, [0, 1, 2]]
                        tris2 = quads[:, [0, 2, 3]]
                        all_indices_list.append(tris1)
                        all_indices_list.append(tris2)
                    else:
                        # Mixed: fallback to slightly faster loop
                        idx_ptr = 0
                        for count in f_counts:
                            if count == 3:
                                all_indices_list.append([f_indices[idx_ptr]+offset, 
                                                       f_indices[idx_ptr+1]+offset, 
                                                       f_indices[idx_ptr+2]+offset])
                            elif count == 4:
                                all_indices_list.append([f_indices[idx_ptr]+offset, 
                                                       f_indices[idx_ptr+1]+offset, 
                                                       f_indices[idx_ptr+2]+offset])
                                all_indices_list.append([f_indices[idx_ptr]+offset, 
                                                       f_indices[idx_ptr+2]+offset, 
                                                       f_indices[idx_ptr+3]+offset])
                            idx_ptr += count
        
        if all_pts_list:
            self.vertices = np.vstack(all_pts_list)
            # Efficiently stack indices
            if all_indices_list:
                self.indices = np.vstack([np.array(i).reshape(-1, 3) for i in all_indices_list])
            
            # =============================================
            # STEP 1: Detect coordinate system (Y-up vs Z-up)
            # =============================================
            # Most USD files from Blender/Maya use Y-up
            # Check if the model extent in Y is larger than Z (typical for Y-up models)
            
            raw_mins = self.vertices.min(axis=0)
            raw_maxs = self.vertices.max(axis=0)
            extent = raw_maxs - raw_mins
            
            # If Y extent is significantly larger than Z, assume Y-up and swap
            if extent[1] > extent[2] * 1.5:
                print("USD: Detected Y-up coordinate system, converting to Z-up...")
                # Swap Y and Z columns
                self.vertices[:, [1, 2]] = self.vertices[:, [2, 1]]
                # Flip the new Z to correct orientation
                self.vertices[:, 2] = -self.vertices[:, 2]
            
            # =============================================
            # STEP 2: Calculate new bounds after axis swap
            # =============================================
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            extent = maxs - mins
            
            print(f"USD bounds: X=[{mins[0]:.2f}, {maxs[0]:.2f}], Y=[{mins[1]:.2f}, {maxs[1]:.2f}], Z=[{mins[2]:.2f}, {maxs[2]:.2f}]")
            
            # =============================================
            # STEP 3: Scale FIRST (so offsets are in final units)
            # =============================================
            max_dim = np.max(extent)
            scale = 5.0 / max_dim if max_dim > 0 else 1.0
            self.vertices *= scale
            
            # =============================================
            # STEP 4: Recalculate bounds after scaling
            # =============================================
            mins = self.vertices.min(axis=0)
            maxs = self.vertices.max(axis=0)
            
            # =============================================
            # STEP 5: Center XY, put Z-bottom at 0
            # =============================================
            center_x = (mins[0] + maxs[0]) / 2.0
            center_y = (mins[1] + maxs[1]) / 2.0
            z_bottom = mins[2]
            
            self.vertices[:, 0] -= center_x
            self.vertices[:, 1] -= center_y
            self.vertices[:, 2] -= z_bottom  # This makes min Z = 0
            
            # =============================================
            # STEP 6: Final bounds check
            # =============================================
            final_mins = self.vertices.min(axis=0)
            final_maxs = self.vertices.max(axis=0)
            self.aabb = AABB(final_mins, final_maxs)
            
            print(f"USD Model positioned: {len(self.vertices)} vertices")
            print(f"  Final Z range: [{final_mins[2]:.4f}, {final_maxs[2]:.2f}] (bottom should be ~0)")
            
            # =============================================
            # STEP 7: Calculate normals for VBO
            # =============================================
            v0 = self.vertices[self.indices[:, 0]]
            v1 = self.vertices[self.indices[:, 1]]
            v2 = self.vertices[self.indices[:, 2]]
            
            # Face normals
            normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms!=0)
            
            # Flatten for VBO: Repeat each normal 3 times for each vertex of the triangle
            self.vbo_data = np.stack([v0, v1, v2], axis=1).reshape(-1, 3)
            self.nbo_data = np.repeat(normals, 3, axis=0)

    def set_lod(self, level):
        """Sets the LOD level (0, 1, 2)"""
        self.lod_level = level
        print(f"USD LOD set to {level}")

    def swap_axes(self):
        """Swaps Y and Z axes (Useful for USD Y-up vs standard Z-up)"""
        if len(self.vbo_data) > 0:
            # Swap Column 1 and 2 (Y and Z)
            self.vbo_data[:, [1, 2]] = self.vbo_data[:, [2, 1]]
            # Also swap normals
            self.nbo_data[:, [1, 2]] = self.nbo_data[:, [2, 1]]
            print("USD Axes Swapped (Y <-> Z)")

    def optimize_geometry(self):
        """Simulates geometry optimization (simplified)"""
        if len(self.vbo_data) > 100000:
            print("Force reduced LOD for huge model")
            self.lod_level = 2

    def render_self(self):
        if not HAS_USD or len(self.vertices) == 0:
            return

        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.8, 0.9, 1.0, 1.0])
        glMaterialfv(GL_FRONT, GL_SHININESS, [100.0])
        glColor3f(0.2, 0.2, 0.25)
        
        # OPTIMIZED RENDERING
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        
        glVertexPointer(3, GL_FLOAT, 0, self.vbo_data)
        glNormalPointer(GL_FLOAT, 0, self.nbo_data)
        
        # Calculate count based on LOD
        total_verts = len(self.vbo_data)
        if self.lod_level == 0:
            glDrawArrays(GL_TRIANGLES, 0, total_verts)
        elif self.lod_level == 1:
            glDrawArrays(GL_TRIANGLES, 0, total_verts // 2)
        else:
            glDrawArrays(GL_TRIANGLES, 0, total_verts // 4)
        
        # Wireframe
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.0, 0.8, 0.5)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        if self.lod_level == 0:
            glDrawArrays(GL_TRIANGLES, 0, total_verts)
        else:
             # Fast wireframe for low LOD
             glDrawArrays(GL_TRIANGLES, 0, total_verts // 10)
             
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glEnable(GL_LIGHTING)

class SnowFigure(HierarchicalNode):
    def __init__(self):
        super().__init__()
        s1 = Sphere(); s1.translate(0, -0.6, 0)
        s2 = Sphere(); s2.translate(0, 0.1, 0); s2.scale(0.8)
        s3 = Sphere(); s3.translate(0, 0.75, 0); s3.scale(0.6)
        self.child_nodes = [s1, s2, s3]
        self.aabb = AABB([-0.5, -1.1, -0.5], [0.5, 1.1, 0.5])

# --- SCENE MANAGER ---
class Scene:
    def __init__(self):
        self.node_list = []
        self.selected_node = None
        self.PLACE_DEPTH = 15.0

    def add_node(self, node):
        self.node_list.append(node)

    def render(self):
        for node in self.node_list:
            node.render()

    def pick(self, start, direction, mat):
        if self.selected_node:
            self.selected_node.selected = False
            self.selected_node = None
        
        mindist = float('inf')
        closest = None
        for node in self.node_list:
            hit, dist = node.pick(start, direction, mat)
            if hit and dist < mindist:
                mindist = dist
                closest = node
        
        if closest:
            closest.selected = True
            closest.depth = mindist
            self.selected_node = closest
        return closest

# --- 3D VIEWER WIDGET ---
class ModelViewer3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = Scene()
        self.last_pos = QPoint()
        self.camera_rot = [25, 45, 0] # pitch, yaw, roll
        self.camera_trans = [0, 0, -15]
        
        # Initial Scene (Ship Hull + Grid)
        self.scene.add_node(ShipHull())
        
        # Real-time Animation Timer (Omniverse Feel)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16) # 60 FPS

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0) # Key Light
        glEnable(GL_LIGHT1) # Fill Light
        glEnable(GL_LIGHT2) # Rim Light
        glEnable(GL_NORMALIZE) # Fixes normals when scaled
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)
        
        # 3-Point Studio Lighting Setup
        glLightfv(GL_LIGHT0, GL_POSITION, [20.0, 30.0, 20.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.12, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.85, 0.9, 1.0])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])

        glLightfv(GL_LIGHT1, GL_POSITION, [-25.0, 5.0, 20.0, 0.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.35, 0.4, 0.5, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])

        glLightfv(GL_LIGHT2, GL_POSITION, [0.0, 40.0, -30.0, 0.0])
        glLightfv(GL_LIGHT2, GL_DIFFUSE, [0.7, 0.8, 1.0, 1.0])
        glLightfv(GL_LIGHT2, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        
        glClearColor(0.04, 0.06, 0.08, 1.0) # Dark industrial theme

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(40, w / h if h > 0 else 1, 0.1, 200.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Camera Transformations
        glTranslatef(*self.camera_trans)
        glRotatef(self.camera_rot[0], 1, 0, 0)
        glRotatef(self.camera_rot[1], 0, 1, 0)
        
        # 1. Digital Ocean Plane
        self.draw_water_plane()
        
        # 2. Main Grid
        self.draw_grid()
        
        # 3. Render Scene (Ships, Objects)
        self.scene.render()

    def draw_water_plane(self):
        """Draws a semi-transparent 'Digital Ocean' plane"""
        glDisable(GL_LIGHTING)
        
        # Water Surface
        glColor4f(0.0, 0.3, 0.5, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(-50, -0.05, -50); glVertex3f(50, -0.05, -50)
        glVertex3f(50, -0.05, 50); glVertex3f(-50, -0.05, 50)
        glEnd()
        
        # Horizon Line
        glLineWidth(1.0)
        glColor4f(0.0, 1.0, 1.0, 0.3)
        glBegin(GL_LINES)
        for i in range(-50, 51, 5):
            glVertex3f(i, -0.05, -50); glVertex3f(i, -0.05, 50)
            glVertex3f(-50, -0.05, i); glVertex3f(50, -0.05, i)
        glEnd()
        
        glEnable(GL_LIGHTING)

    def draw_grid(self):
        glDisable(GL_LIGHTING)
        glLineWidth(0.5)
        glBegin(GL_LINES)
        glColor3f(0.15, 0.15, 0.2)
        for i in range(-20, 21):
            glVertex3f(i, -0.1, -20); glVertex3f(i, -0.1, 20)
            glVertex3f(-20, -0.1, i); glVertex3f(20, -0.1, i)
        glEnd()
        glEnable(GL_LIGHTING)

    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            # Picking logic
            self.perform_pick(event.position().x(), event.position().y())
        self.update()

    def mouseMoveEvent(self, event):
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        
        if event.buttons() & Qt.MouseButton.RightButton:
            self.camera_rot[0] += dy
            self.camera_rot[1] += dx
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self.camera_trans[0] += dx * 0.05
            self.camera_trans[1] -= dy * 0.05
            
        self.last_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        self.camera_trans[2] += event.angleDelta().y() / 120.0
        self.update()

    def perform_pick(self, x, y):
        # Convert screen coords to ray (simplified)
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Near point
        p_near = gluUnProject(x, viewport[3] - y, 0.0, modelview, projection, viewport)
        # Far point
        p_far = gluUnProject(x, viewport[3] - y, 1.0, modelview, projection, viewport)
        
        start = np.array(p_near)
        direction = np.array(p_far) - start
        direction /= np.linalg.norm(direction)
        
        # Invert modelview to get scene-to-world
        inv_mv = np.linalg.inv(modelview.T)
        self.scene.pick(start, direction, inv_mv)

    def add_shape(self, shape_type):
        if shape_type == "cube": node = Cube()
        elif shape_type == "sphere": node = Sphere()
        elif shape_type == "figure": node = SnowFigure()
        else: return
        
        node.translate(0, 0, 0)
        self.scene.add_node(node)
        self.update()

    def update_cfd_results(self, results):
        """Adds or updates the CFD visualization in the 3D scene"""
        # Remove old CFD nodes if any
        self.scene.node_list = [n for n in self.scene.node_list if not isinstance(n, CFDNode)]
        
        # Add new CFD Node
        cfd_node = CFDNode(results)
        self.scene.add_node(cfd_node)
        self.update()

    def update_vessel_hull(self, vessel_data):
        """Updates the 3D ship hull based on technical parameters (LOA, Beam, Draft)"""
        # Remove old hull nodes
        self.scene.node_list = [n for n in self.scene.node_list if not isinstance(n, ShipHull)]
        
        # Create new hull with scaled dimensions
        loa = vessel_data.get('loa', 190.0)
        lbp = vessel_data.get('lbp', 182.0)
        beam = vessel_data.get('beam', 32.2)
        draft = vessel_data.get('draft', 12.5)
        depth = vessel_data.get('depth', 18.0)
        cb = vessel_data.get('cb', 0.82)
        cp = vessel_data.get('cp', 0.84)
        cm = vessel_data.get('cm', 0.98)
        bow_h = vessel_data.get('bow_height', 4.0)
        stern_h = vessel_data.get('stern_height', 2.0)
        bulb_l = vessel_data.get('bulb_length', 5.5)
        bulb_r = vessel_data.get('bulb_radius', 2.2)
        stern_s = vessel_data.get('stern_shape', 0.8)
        prop_d = vessel_data.get('prop_dia', 6.5)
        prop_b = vessel_data.get('prop_blades', 4)
        rudder_h = vessel_data.get('rudder_h', 7.0)
        v_type = vessel_data.get('type', "Bulk Carrier")
        retrofits = vessel_data.get('selected_retrofit', [])
        
        hull = ShipHull(loa=loa, lbp=lbp, beam=beam, draft=draft, depth=depth, 
                        cb=cb, cp=cp, cm=cm, bow_height=bow_h, stern_height=stern_h,
                        bulb_l=bulb_l, bulb_r=bulb_r, stern_s=stern_s,
                        prop_d=prop_d, prop_b=prop_b, rudder_h=rudder_h,
                        vessel_type=v_type, retrofit_components=retrofits)
        self.scene.add_node(hull)
        self.update()

    def load_stl(self, file_path):
        """Loads and displays an STL model"""
        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            return
            
        # Remove old ship/stl nodes
        self.scene.node_list = [n for n in self.scene.node_list 
                               if not isinstance(n, (ShipHull, STLHull))]
        
        try:
            stl_node = STLHull(file_path)
            self.scene.add_node(stl_node)
            self.update()
            print(f"STL Loaded: {file_path}")
        except Exception as e:
            print(f"Failed to load STL: {e}")

    def load_usd(self, file_path):
        """Loads and displays a USD model"""
        if not HAS_USD:
            print("Error: pxr (usd-core) package is required for USD files.")
            return

        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            return
            
        # Remove old ship/stl/usd nodes
        self.scene.node_list = [n for n in self.scene.node_list 
                               if not isinstance(n, (ShipHull, STLHull, USDHull))]
        
        try:
            usd_node = USDHull(file_path)
            self.scene.add_node(usd_node)
            self.update()
            print(f"USD Loaded: {file_path}")
        except Exception as e:
            print(f"Failed to load USD: {e}")
