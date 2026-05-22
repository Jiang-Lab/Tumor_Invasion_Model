from cc3d.cpp.PlayerPython import *
from cc3d import CompuCellSetup
from cc3d.core.PySteppables import *

from datetime import datetime
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.signal import find_peaks
import os, csv, math, random

# =========================
# Global knobs
# =========================
k = 25          # (%) leaders target fraction in the blob
fgrow = 0.015   # follower growth increment per MCS (until cap)
a = 4           # Cell Width
# vol0 = (math.pi / 6.0) * (a ** 3)        initial target volume per cell (lattice voxels)
vol0 = 25
Jlf = 1         # Contact energy LC–FC (overwrites XML id="J_LF")
mu  = 30        # Chemotaxis lambda for LC (overwrites XML id="lambda_chem")
PP  = 0.9       # Prob. follower is proliferative (has division clock)

show_plots = 0  # CC3D built-in plotting (2D only) -> keep 0 for 3D runs

# ======================================================================
# ConstraintInitializer: 3D radial chemotaxis + enforce k% LC in sphere
# ======================================================================
class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        super().__init__(frequency)

    def start(self):
        # CSV of LC/FC counts
        self.cellcount_file = open(os.path.join(self.output_dir, f"CellCount3D_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.cellcount_writer = csv.writer(self.cellcount_file)
        self.cellcount_writer.writerow(["MCS", "Leader Cells", "Follower Cells", "Total"])

        # Ensure k% LC inside the FC blob seeded by BlobInitializer (XML)
        def frac_leaders():
            lc = len(self.cell_list_by_type(self.LC))
            tot = lc + len(self.cell_list_by_type(self.FC))
            return 0.0 if tot == 0 else lc / tot

        # Convert random FC voxels to new LC cells until we reach k%
    
        while frac_leaders() < k / 100.0:
            x1 = np.random.randint(0, self.dim.x)
            y1 = np.random.randint(0, self.dim.y)
            z1 = np.random.randint(0, self.dim.z)
            c = self.cellField[x1, y1, z1]
            if c and c.type == self.FC:
                lc = self.new_cell(self.LC)
                self.cellField[x1, y1, z1] = lc

        # Starting volume constraints
        for cell in self.cell_list_by_type(self.FC, self.LC):
            cell.targetVolume = vol0
            cell.lambdaVolume = 2.0

        # Overwrite XML params (adhesion + chemotaxis lambda)
        self.get_xml_element("J_LF").cdata = Jlf
        self.get_xml_element("lambda_chem").Lambda = mu

        # ----- 3D RADIAL CHEMOTAXIS FIELD (MV = distance from center) -----
        mv = self.field.MV
        cx, cy, cz = self.dim.x // 2, self.dim.y // 2, self.dim.z // 2
        g = 1.0  # increase g to weaken gradient
        
        Rmax = min(cx, cy, cz)
        if Rmax == 0:
            Rmax = 1  
        for x in range(self.dim.x):
            dx = x - cx
            for y in range(self.dim.y):
                dy = y - cy
                for z in range(self.dim.z):
                    dz = z - cz
                    r = math.sqrt(dx*dx + dy*dy + dz*dz)
                    mv[x, y, z] = r / g
                    # mv[x, y, z] = min(1.0, r / Rmax) 

    def step(self, mcs):
        final_step = self.simulator.getNumSteps()
        L = len(self.cell_list_by_type(self.LC))
        F = len(self.cell_list_by_type(self.FC))
        T = len(self.cell_list)
        self.cellcount_writer.writerow([mcs, L, F, T])
        if mcs == final_step - 1:
            self.cellcount_file.close()

# ======================================================================
# Growth: followers grow until ~2*vol0 (divide threshold)
# ======================================================================
class GrowthSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        super().__init__(frequency)

    def step(self, mcs):
        for cell in self.cell_list_by_type(self.FC):
            if cell.targetVolume < 2 * vol0:
                cell.targetVolume += fgrow

# ======================================================================
# Mitosis: followers divide if clock & volume threshold reached
# ======================================================================
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        super().__init__(frequency)

    def start(self):
        for cell in self.cell_list_by_type(self.FC):
            if random.random() <= PP:
                cell.dict["clock"] = np.random.randint(0, 75)
            else:
                cell.dict["clock"] = None

    def step(self, mcs):
        to_divide = []
        for cell in self.cell_list_by_type(self.FC):
            clk = cell.dict.get("clock", None)
            if clk is not None:
                cell.dict["clock"] = clk + 1
                vary = np.random.randint(0, 50)
                if cell.volume > 2 * vol0 and cell.dict["clock"] > 75 + vary:
                    to_divide.append(cell)
        for c in to_divide:
            # Divide along a random orientation in 3D
            self.divide_cell_random_orientation(c)

    def update_attributes(self):
        self.parent_cell.targetVolume /= 2.0
        self.parent_cell.dict["clock"] = 0
        self.clone_parent_2_child()
        # followers stay followers on division
        if self.parent_cell.type == self.FC:
            self.child_cell.type = self.FC
        else:
            self.child_cell.type = self.LC

# ======================================================================
# Analysis & tracking (3D)
# ======================================================================
class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency)
        # cluster ID tracking
        self.prev_tracks = []
        self.next_cluster_id = 1
        self.centroid_match_radius = 14.0
        self.prev_clusters_info = []

    # ---------- lifecycle ----------
    def start(self):
        self._init_outputs()
        self.create_scalar_field_cell_level_py("myField")  # same as before; can be used for tips if desired

    def step(self, mcs):
        final_step = self.simulator.getNumSteps()

        # main tumor: BFS from the center voxel
        Tumor = nx.Graph()
        surface = []
        Tumorcells = self._collect_main_tumor_3d(Tumor, surface)  # set of IDs

        # baseline inner radius
        baseline_r = self._baseline_radius(Tumorcells)

        # single LC with no neighbors and at r > baseline
        defectorcells = self._find_single_leader_defectors(baseline_r)

        # detached (neither main tumor nor adjacent to it), at r > baseline
        detached_cells = self._find_detached_non_tumor_cells(Tumorcells, baseline_r)
        num_detached = len(detached_cells)

        # clusters (non-tumor FC-based clusters)
        clustercells = []
        cluster_compositions = self._collect_clusters_outside_tumor(Tumor, Tumorcells, clustercells)

        # stable IDs across time
        if cluster_compositions:
            cluster_compositions.sort(key=lambda c: (c["centroid_x"], c["centroid_y"], c["centroid_z"]))
            for i, c in enumerate(cluster_compositions, start=1):
                c["ClusterID_frame"] = i
            self._assign_stable_ids(cluster_compositions)

        curr_member_union = set().union(*(set(c["member_ids"]) for c in cluster_compositions)) \
            if cluster_compositions else set()
        singleton_ids = set(detached_cells) - curr_member_union

        merges, splits, dissolves, curr_info = self._detect_merge_split(
            self.prev_clusters_info,
            cluster_compositions,
            curr_tumor_ids=set(Tumorcells),
            singleton_ids=singleton_ids
        )
        for evt in merges:
            self.events_writer.writerow([mcs, "merge", evt["parents"], [evt["child"]], evt["fractions"], "", ""])
        for evt in splits:
            self.events_writer.writerow([mcs, "split", [evt["parent"]], evt["children"], evt["fractions"], "", ""])
        for evt in dissolves:
            self.events_writer.writerow([
                mcs, "dissolve", [evt["parent"]], [], [],
                round(evt["lost_to_singletons"], 3),
                round(evt["lost_to_main_tumor"], 3)
            ])
        self.prev_clusters_info = curr_info

        if mcs % 10 == 0:
            self.save_cluster_compositions(cluster_compositions, mcs)

        # 3D radial boundaries by ray-marching over angles (θ, φ)
        theta, phi, r_main, r_outer, r_base = self._scan_boundaries_spherical(Tumorcells, clustercells,
                                                                              n_theta=96, n_phi=48)

        # Invasive & infiltrative volumes via spherical sector: V = ∫ (r^3 / 3) dΩ
        inv_vol, inf_vol = self._volumes_spherical(theta, phi, r_main, r_outer, r_base)

        # save metrics row
        self.metrics_writer.writerow([mcs, inv_vol, inf_vol,
                                      len(defectorcells), num_detached, len(cluster_compositions)])

        # periodic dumps
        if mcs % 100 == 0 and r_main.size:
            boundary_data = []
            for i in range(len(r_main)):
                boundary_data.append([float(theta[i]), float(phi[i]),
                                      float(r_main[i]), float(r_outer[i]), float(r_base[i])])

            defector_positions = self._positions(detached_cells)
            tumor_positions, tumor_LC, tumor_FC = self._tumor_positions(Tumorcells)

            self.save_mcs_data(self.output_dir1, mcs,
                               boundary_data,
                               defector_positions,
                               tumor_positions,
                               tumor_LC,
                               tumor_FC,
                               spherical=True)

        # final text log
        if mcs == final_step - 1:
            # perimeter/complexity notions are 2D; here we just summarize counts/volumes.
            self._final_text_log(mcs, inv_vol, inf_vol,
                                 len(defectorcells), num_detached,
                                 len(cluster_compositions),
                                 cluster_compositions)

    def finish(self):
        try:
            self.events_file.close()
            self.metrics_file.close()
            self.f.close()
        except Exception:
            pass

    # ---------- outputs ----------
    def _init_outputs(self):
        self.output_dir1 = self.output_dir + f"/PositionData3D_{Jlf}_{mu}_{PP}"
        os.makedirs(self.output_dir1, exist_ok=True)

        self.metrics_file = open(os.path.join(self.output_dir, f"Metrics3D_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(["MCS", "InvasiveVolume", "InfiltrativeVolume", "SingleDefects", "Detached", "Clusters"])

        self.events_file = open(os.path.join(self.output_dir, f"ClusterEvents3D_{Jlf}_{mu}_{PP}.csv"), "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(["MCS","Event","Parents","Children","OverlapFractions","LostToSingletons","LostToMainTumor"])

        self.f = open(self.output_dir + f"/summary3D_{Jlf}_{mu}_{PP}.txt", "a")

    def save_cluster_compositions(self, clusters_data, mcs):
        mcs_folder = os.path.join(self.output_dir1, f"MCS_{mcs}")
        os.makedirs(mcs_folder, exist_ok=True)
        fn = os.path.join(mcs_folder, f"ClusterComposition3D_{Jlf}_{mu}_{PP}_MCS_{mcs}.csv")
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ClusterID_stable","ClusterID_frame",
                "LC_count","FC_count","Total",
                "Centroid_X","Centroid_Y","Centroid_Z"
            ])
            for c in clusters_data:
                w.writerow([
                    c.get("ClusterID_stable"), c.get("ClusterID_frame"),
                    c["leader_cells"], c["follower_cells"], c["total_cells"],
                    c["centroid_x"], c["centroid_y"], c["centroid_z"]
                ])

    def save_mcs_data(self, output_dir1, mcs,
                      boundary_data,
                      defector_positions,
                      tumor_positions,
                      tumor_leader_cells,
                      tumor_follower_cells,
                      spherical=True):
        mcs_folder = os.path.join(output_dir1, f"MCS_{mcs}")
        os.makedirs(mcs_folder, exist_ok=True)
        tag = f"{Jlf}_{mu}_{PP}_MCS_{mcs}"

        def dump(name, header, rows):
            path = os.path.join(mcs_folder, f"{name}_{tag}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)

        if boundary_data:
            if spherical:
                dump("BoundaryData3D",
                     ["Theta","Phi","R_Main","R_Outer","R_Base"], boundary_data)
            else:
                dump("BoundaryData", ["X","Main","Outer","Base"], boundary_data)

        if defector_positions:
            dump("DefectorPosition", ["id","x","y","z"], defector_positions)
        if tumor_positions:
            dump("TumorPosition", ["id","x","y","z"], tumor_positions)
        if tumor_leader_cells:
            dump("TumorLeaderCells", ["id","x","y","z"], tumor_leader_cells)
        if tumor_follower_cells:
            dump("TumorFollowerCells", ["id","x","y","z"], tumor_follower_cells)

    # ---------- helpers: BFS & tumor / cluster discovery ----------
    def _bfs(self, Tumor, surface, start_cell):
        """BFS over 3D neighbors; build a graph of IDs with edge weights."""
        visited = set([start_cell.id])
        q = [start_cell.id]
        Tumor.add_node(start_cell.id, xCOM=start_cell.xCOM, yCOM=start_cell.yCOM, zCOM=start_cell.zCOM)
        while q:
            mid = q.pop(0)
            m = self.fetch_cell_by_id(mid)
            for neigh, common_surf in self.get_cell_neighbor_data_list(m):
                if neigh:
                    if neigh.id not in visited:
                        visited.add(neigh.id)
                        q.append(neigh.id)
                        Tumor.add_node(neigh.id, xCOM=neigh.xCOM, yCOM=neigh.yCOM, zCOM=neigh.zCOM)
                    dx = m.xCOM - neigh.xCOM
                    dy = m.yCOM - neigh.yCOM
                    dz = m.zCOM - neigh.zCOM
                    w = math.sqrt(dx*dx + dy*dy + dz*dz)
                    Tumor.add_edge(m.id, neigh.id, weight=w)
                if not neigh:
                    if m.id not in surface:
                        surface.append(m.id)
        return visited

    def _center(self):
        return self.dim.x // 2, self.dim.y // 2, self.dim.z // 2

    def _collect_main_tumor_3d(self, Tumor, surface):
        """Seed BFS from nearest cell to the center voxel."""
        cx, cy, cz = self._center()
        start = self.cell_field[cx, cy, cz]
        if not start:
            # expand cube shells until we find a cell
            max_r = max(self.dim.x, self.dim.y, self.dim.z)
            found = None
            for r in range(1, max_r):
                # sample the 6 faces of the cube shell (fast and sufficient)
                for x in (cx-r, cx+r):
                    for y in range(cy-r, cy+r+1):
                        for z in range(cz-r, cz+r+1):
                            if 0 <= x < self.dim.x and 0 <= y < self.dim.y and 0 <= z < self.dim.z:
                                c = self.cell_field[x, y, z]
                                if c:
                                    found = c; break
                        if found: break
                    if found: break
                if not found:
                    for y in (cy-r, cy+r):
                        for x in range(cx-r, cx+r+1):
                            for z in range(cz-r, cz+r+1):
                                if 0 <= x < self.dim.x and 0 <= y < self.dim.y and 0 <= z < self.dim.z:
                                    c = self.cell_field[x, y, z]
                                    if c:
                                        found = c; break
                            if found: break
                        if found: break
                if not found:
                    for z in (cz-r, cz+r):
                        for x in range(cx-r, cx+r+1):
                            for y in range(cy-r, cy+r+1):
                                if 0 <= x < self.dim.x and 0 <= y < self.dim.y and 0 <= z < self.dim.z:
                                    c = self.cell_field[x, y, z]
                                    if c:
                                        found = c; break
                            if found: break
                        if found: break
                if found:
                    start = found
                    break
        if not start:
            return set()
        return self._bfs(Tumor, surface, start)

    def _center_radius(self, cell):
        cx, cy, cz = self._center()
        dx = cell.xCOM - cx; dy = cell.yCOM - cy; dz = cell.zCOM - cz
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _baseline_radius(self, Tumorcells):
        rmin = float("inf")
        for cid in Tumorcells:
            c = self.fetch_cell_by_id(cid)
            if c:
                rmin = min(rmin, self._center_radius(c))
        return 0.0 if rmin == float("inf") else rmin

    def _find_single_leader_defectors(self, baseline_r):
        out = []
        for c in self.cell_list_by_type(self.LC):
            n = 0
            for neigh, _ in self.get_cell_neighbor_data_list(c):
                if neigh: n += 1
            if n == 0 and self._center_radius(c) > baseline_r:
                out.append(c.id)
        return out

    def _find_detached_non_tumor_cells(self, Tumorcells, baseline_r):
        out = []
        T = set(Tumorcells)
        for c in self.cell_list_by_type(self.LC, self.FC):
            if c.id in T: 
                continue
            neighs = [n for n,_ in self.get_cell_neighbor_data_list(c) if n]
            if all(n.id not in T for n in neighs):
                if self._center_radius(c) > baseline_r:
                    out.append(c.id)
        return out

    def _collect_clusters_outside_tumor(self, Tumor, Tumorcells, clustercells):
        checked = set()
        comps = []
        T = set(Tumorcells)
        for c in self.cell_list_by_type(self.FC):
            if c.id in T or c.id in checked:
                continue
            newset = self._bfs(Tumor, [], c)
            if len(newset) >= 2:
                checked.update(newset)
                clustercells.extend(newset)
                # counts
                lc_cnt = sum(1 for i in newset if self.fetch_cell_by_id(i).type == self.LC)
                fc_cnt = sum(1 for i in newset if self.fetch_cell_by_id(i).type == self.FC)
                # centroid
                xs, ys, zs = [], [], []
                for i in newset:
                    ci = self.fetch_cell_by_id(i)
                    if ci:
                        xs.append(ci.xCOM); ys.append(ci.yCOM); zs.append(ci.zCOM)
                cx = float(np.mean(xs)) if xs else 0.0
                cy = float(np.mean(ys)) if ys else 0.0
                cz = float(np.mean(zs)) if zs else 0.0
                comps.append({
                    "leader_cells": lc_cnt,
                    "follower_cells": fc_cnt,
                    "total_cells": len(newset),
                    "centroid_x": cx, "centroid_y": cy, "centroid_z": cz,
                    "member_ids": list(newset)
                })
        return comps

    # ---------- spherical boundary scan ----------
    def _scan_boundaries_spherical(self, Tumorcells, clustercells, n_theta=96, n_phi=48):
        """
        Directions are parameterized by (theta, phi):
          theta ∈ [0, 2π)   (azimuth)
          phi   ∈ [0, π]    (polar angle from +Z axis)
        For each (theta, phi), ray-march from center, record last radius
        intersecting main tumor (r_main) and any tissue (r_outer).
        """
        cx, cy, cz = self._center()
        T = set(Tumorcells)
        C = set(clustercells)

        thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
        phis   = np.linspace(1e-6, math.pi - 1e-6, n_phi, endpoint=True)  # avoid exact poles for stable stepping

        rmax = math.sqrt(self.dim.x**2 + self.dim.y**2 + self.dim.z**2)
        rs_main, rs_outer, th_list, ph_list = [], [], [], []

        for phi in phis:
            sp = math.sin(phi)
            cp = math.cos(phi)
            for th in thetas:
                ct = math.cos(th)
                st = math.sin(th)
                # direction vector
                vx, vy, vz = (sp * ct, sp * st, cp)

                r = 0.0
                step = 1.0  # voxel step; reduce to 0.5 for more accuracy
                last_main = 0.0
                last_outer = 0.0
                hit_main = False
                hit_outer = False

                while r < rmax:
                    x = int(round(cx + r * vx))
                    y = int(round(cy + r * vy))
                    z = int(round(cz + r * vz))
                    if 0 <= x < self.dim.x and 0 <= y < self.dim.y and 0 <= z < self.dim.z:
                        cell = self.cell_field[x, y, z]
                        if cell:
                            hit_outer = True
                            last_outer = r
                            if (cell.id in T) and (cell.id not in C):
                                hit_main = True
                                last_main = r
                    else:
                        break
                    r += step

                th_list.append(th)
                ph_list.append(phi)
                rs_main.append(last_main if hit_main else 0.0)
                rs_outer.append(last_outer if hit_outer else 0.0)

        th_arr = np.array(th_list, dtype=float)
        ph_arr = np.array(ph_list, dtype=float)
        r_main = np.array(rs_main, dtype=float)
        r_outer = np.array(rs_outer, dtype=float)

        # baseline = min positive r_main across all directions
        pos = r_main[r_main > 0]
        r_base = np.zeros_like(r_main)
        if pos.size:
            r_base[:] = float(np.min(pos))
        return th_arr, ph_arr, r_main, r_outer, r_base

    def _solid_angle_weights(self, theta, phi):
        """
        Approximate ΔΩ weights for each (θ, φ) cell in a regular grid:
        ΔΩ ≈ Δθ * Δφ * sin(φ)
        """
        # infer Δθ and Δφ from unique sorted lists
        thetas = np.unique(theta)
        phis = np.unique(phi)
        if thetas.size > 1:
            dtheta = float(np.min(np.diff(thetas)))
        else:
            dtheta = 2.0 * math.pi
        if phis.size > 1:
            dphi = float(np.min(np.diff(phis)))
        else:
            dphi = math.pi

        return dtheta * dphi * np.sin(phi)

    def _volumes_spherical(self, theta, phi, r_main, r_outer, r_base):
        """
        Volume of a spherical sector = (r^3 / 3) ΔΩ.
        We integrate:
          invasive ≈ Σ ((max(r_main, r_base)^3 - r_base^3)/3 * ΔΩ)
          infiltrative ≈ Σ ((max(r_outer, r_base)^3 - r_base^3)/3 * ΔΩ)
        """
        dOmega = self._solid_angle_weights(theta, phi)
        rm3 = np.maximum(r_main, r_base)**3 - r_base**3
        ro3 = np.maximum(r_outer, r_base)**3 - r_base**3
        invasive = np.sum((rm3 / 3.0) * dOmega)
        infiltrative = np.sum((ro3 / 3.0) * dOmega)
        return float(invasive), float(infiltrative)

    # ---------- periodic helpers ----------
    def _positions(self, ids):
        out = []
        for cid in ids:
            c = self.fetch_cell_by_id(cid)
            if c:
                out.append([c.id, c.xCOM, c.yCOM, c.zCOM])
        return out

    def _tumor_positions(self, Tumorcells):
        tumor_positions, tumor_LC, tumor_FC = [], [], []
        for cid in Tumorcells:
            c = self.fetch_cell_by_id(cid)
            if c:
                tumor_positions.append([c.id, c.xCOM, c.yCOM, c.zCOM])
                if c.type == self.LC:
                    tumor_LC.append([c.id, c.xCOM, c.yCOM, c.zCOM])
                elif c.type == self.FC:
                    tumor_FC.append([c.id, c.xCOM, c.yCOM, c.zCOM])
        return tumor_positions, tumor_LC, tumor_FC

    # ---------- stable IDs & events ----------
    def _euclid(self, a, b):
        dx = a[0]-b[0]; dy = a[1]-b[1]; dz = a[2]-b[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _assign_stable_ids(self, comps):
        if not comps:
            return
        if not self.prev_tracks:
            for c in sorted(comps, key=lambda c: (c["centroid_x"], c["centroid_y"], c["centroid_z"])):
                c["ClusterID_stable"] = self.next_cluster_id
                self.prev_tracks.append({
                    "id": self.next_cluster_id,
                    "centroid": (float(c["centroid_x"]), float(c["centroid_y"]), float(c["centroid_z"]))
                })
                self.next_cluster_id += 1
            return

        used_prev = set()
        new_tracks = []
        for c in comps:
            cen = (float(c["centroid_x"]), float(c["centroid_y"]), float(c["centroid_z"]))
            best = None; best_d = float("inf")
            for tr in self.prev_tracks:
                if tr["id"] in used_prev: continue
                d = self._euclid(cen, tr["centroid"])
                if d < best_d:
                    best_d = d; best = tr
            if best is not None and best_d <= self.centroid_match_radius:
                c["ClusterID_stable"] = best["id"]
                used_prev.add(best["id"])
                new_tracks.append({"id": best["id"], "centroid": cen})
            else:
                cid = self.next_cluster_id
                c["ClusterID_stable"] = cid
                new_tracks.append({"id": cid, "centroid": cen})
                self.next_cluster_id += 1
        self.prev_tracks = new_tracks

    def _detect_merge_split(self, prev_info, curr_comps, curr_tumor_ids, singleton_ids,
                            jaccard_min=0.15, cover_min=0.25):
        curr_info = [
            {"id": c.get("ClusterID_stable"), "members": set(c.get("member_ids", []))}
            for c in curr_comps if c.get("ClusterID_stable") is not None
        ]

        def jacc(a, b):
            if not a or not b: return 0.0
            inter = len(a & b); uni = len(a | b)
            return 0.0 if uni == 0 else inter / uni

        merges, splits, dissolves = [], [], []

        # merges: many prev -> one curr
        for c in curr_info:
            parents, fracs = [], []
            for p in prev_info:
                J = jacc(p["members"], c["members"])
                cover_child = len(p["members"] & c["members"]) / max(1, len(c["members"]))
                if J >= jaccard_min and cover_child >= cover_min:
                    parents.append(p["id"]); fracs.append(round(cover_child, 3))
            if len(parents) >= 2:
                merges.append({"child": c["id"], "parents": parents, "fractions": fracs})

        # splits / dissolves
        for p in prev_info:
            children, fracs = [], []
            for c in curr_info:
                J = jacc(p["members"], c["members"])
                cover_parent = len(p["members"] & c["members"]) / max(1, len(p["members"]))
                if J >= jaccard_min and cover_parent >= cover_min:
                    children.append(c["id"]); fracs.append(round(cover_parent, 3))
            if len(children) >= 2:
                splits.append({"parent": p["id"], "children": children, "fractions": fracs})
            elif len(children) == 0:
                parent_sz = max(1, len(p["members"]))
                lost_to_singletons = len(p["members"] & singleton_ids) / parent_sz
                lost_to_main = len(p["members"] & curr_tumor_ids) / parent_sz
                dissolves.append({
                    "parent": p["id"],
                    "lost_to_singletons": lost_to_singletons,
                    "lost_to_main_tumor": lost_to_main
                })
        return merges, splits, dissolves, curr_info

    # ---------- final text ----------
    def _final_text_log(self, mcs, inv_vol, inf_vol,
                        single_defects, detached_cells_count,
                        clusters, cluster_compositions):
        self.f.write(f"FINAL STEP {mcs}\n")
        self.f.write(f"InvasiveVolume: {inv_vol}\n")
        self.f.write(f"InfiltrativeVolume: {inf_vol}\n")
        self.f.write(f"SingleDefects: {single_defects}\n")
        self.f.write(f"DetachedCells: {detached_cells_count}\n")
        self.f.write(f"Clusters: {clusters}\n")
        self.f.write("Cluster Composition:\n")
        for c in cluster_compositions:
            self.f.write(
                f"  ID {c.get('ClusterID_stable')} (frame {c.get('ClusterID_frame')}): "
                f"LC={c['leader_cells']} FC={c['follower_cells']} Total={c['total_cells']} | "
                f"Centroid=({c['centroid_x']:.1f},{c['centroid_y']:.1f},{c['centroid_z']:.1f})\n"
            )
        self.f.write("END\n")
        
        
        
        
        
        
# from cc3d.cpp.PlayerPython import *
# from cc3d.core.PySteppables import *
# from cc3d import CompuCellSetup

# import numpy as np, math, os, csv
# from datetime import datetime
# from pathlib import Path
# import networkx as nx
# from scipy.signal import find_peaks

# # ---------- global knobs ----------
# k = 25         # % of LC seeded within the spheroid
# fgrow = 0.015  # FC growth per MCS
# lgrow = 0.010  # (kept for parity; not used unless you want LC growth)
# Jlf  = 1       # LC-FC contact energy override
# mu   = 30      # LC chemotaxis lambda override
# PP   = 0.9     # fraction of FC allowed to proliferate
# vol0 = 20      # initial target volume (lattice voxels)
# pin_to_floor_strength = -0.5  # 0.0 disables; increase (e.g., 1–2) to push cells gently toward −z


# show_plots = 0 # set 1 to enable CC3D plotting windows

# # =========================================================
# # ConstraintInitializerSteppable
# # =========================================================
# class ConstraintInitializerSteppable(SteppableBasePy):
    # def __init__(self, frequency=1):
        # super().__init__(frequency)

    # def start(self):
        # # logging: cell counts
        # self.cellcount_file = open(os.path.join(self.output_dir, f"CellCount_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        # self.cellcount_writer = csv.writer(self.cellcount_file)
        # self.cellcount_writer.writerow(["MCS", "Leader Cells", "Follower Cells", "Total"])

        # # convert FC pixels to LC until k%
        # def frac_leaders():
            # lc = len(self.cell_list_by_type(self.LC))
            # total = lc + len(self.cell_list_by_type(self.FC))
            # return 0.0 if total == 0 else lc / total

        # while frac_leaders() < k/100.0:
            # x = np.random.randint(0, self.dim.x)
            # y = np.random.randint(0, self.dim.y)
            # z = np.random.randint(0, self.dim.z)
            # c = self.cell_field[x, y, z]
            # if c and c.type == self.FC:
                # lc = self.new_cell(self.LC)
                # self.cell_field[x, y, z] = lc

        # # set volumes
        # for cell in self.cell_list_by_type(self.FC, self.LC):
            # cell.targetVolume = vol0
            # cell.lambdaVolume = 2.0

        # # override XML params
        # self.get_xml_element("J_LF").cdata = Jlf
        # self.get_xml_element("lambda_chem").Lambda = mu

        # # build a 3D radial chemotaxis field (
        # mv = self.field.MV
        # cx, cy, cz = self.dim.x//2, self.dim.y//2, self.dim.z//2
        # g = 1.0  # scale
        # for x in range(self.dim.x):
            # dx = x - cx
            # for y in range(self.dim.y):
                # dy = y - cy
                # for z in range(self.dim.z):
                    # dz = z - cz
                    # mv[x, y, z] = (dx*dx + dy*dy + dz*dz) ** 0.5 / g
                    
            # # ----- LATERAL-ONLY CHEMOTAXIS FIELD -----
        # # Leaders will chemotax sideways because MV depends only on x,y (not z).
        ## mv = self.field.MV
        ##cx, cy = self.dim.x // 2, self.dim.y // 2
        ##for x in range(self.dim.x):
            ## dx = x - cx
            ## for y in range(self.dim.y):
                ## dy = y - cy
                ## r_xy = math.hypot(dx, dy)
                ## # Fill entire column along z with the same lateral value
                ## mv[x, y, :] = r_xy

        
        # # Uses ExternalPotential plugin.
        # if pin_to_floor_strength and pin_to_floor_strength > 0.0:
            # for cell in self.cell_list_by_type(self.FC, self.LC):
                # cell.lambdaVecX = 0.0
                # cell.lambdaVecY = 0.0
                # cell.lambdaVecZ = -float(pin_to_floor_strength)            
                    

    # def step(self, mcs):
        # final_step = self.simulator.getNumSteps()
        # L = len(self.cell_list_by_type(self.LC))
        # F = len(self.cell_list_by_type(self.FC))
        # T = len(self.cell_list)
        # self.cellcount_writer.writerow([mcs, L, F, T])
        # if mcs == final_step - 1:
            # self.cellcount_file.close()

# # =========================================================
# # GrowthSteppable
# # =========================================================
# class GrowthSteppable(SteppableBasePy):
    # def __init__(self, frequency=1):
        # super().__init__(frequency)

    # def step(self, mcs):
        # for cell in self.cell_list_by_type(self.FC):
            # if cell.targetVolume < 2 * vol0:
                # cell.targetVolume += fgrow

# # =========================================================
# # MitosisSteppable
# # =========================================================
# class MitosisSteppable(MitosisSteppableBase):
    # def __init__(self, frequency=1):
        # super().__init__(frequency)

    # def start(self):
        # for cell in self.cell_list_by_type(self.FC):
            # cell.dict["clock"] = np.random.randint(0, 75) if np.random.rand() <= PP else None

    # def step(self, mcs):
        # to_divide = []
        # for cell in self.cell_list_by_type(self.FC):
            # clk = cell.dict.get("clock", None)
            # if clk is None: 
                # continue
            # cell.dict["clock"] = clk + 1
            # vary = np.random.randint(0, 50)
            # if cell.volume > 2 * vol0 and cell.dict["clock"] > 75 + vary:
                # to_divide.append(cell)
        # for c in to_divide:
            # self.divide_cell_random_orientation(c)

    # def update_attributes(self):
        # self.parent_cell.targetVolume /= 2.0
        # self.parent_cell.dict["clock"] = 0
        # self.clone_parent_2_child()
        # self.child_cell.type = self.parent_cell.type

# # =========================================================
# # NeighborTrackerPrinterSteppable (3D)
# # =========================================================
# class NeighborTrackerPrinterSteppable(SteppableBasePy):
    # def __init__(self, frequency=10):
        # super().__init__(frequency)
        # self.prev_tracks = []
        # self.next_cluster_id = 1
        # self.centroid_match_radius = 12.0
        # self.prev_clusters_info = []

    # def start(self):
        # self._init_outputs()
        # if show_plots:
            # self._init_plots()
        # self.create_scalar_field_cell_level_py("myField")

    # def step(self, mcs):
        # final_step = self.simulator.getNumSteps()
        # tips = self.field.myField
        # tips.clear()

        # Tumor = nx.Graph()
        # surface = []

        # # main mass (BFS seeded near spheroid center)
        # Tumorcells = self._collect_main_tumor(Tumor, surface)
        # if not Tumorcells:
            # return

        # # baseline radius (closest radius among tumor cells)
        # r_base = self._baseline_radius(Tumorcells)

        # # leader singletons (no neighbors) outside baseline
        # defectorcells = self._find_single_leader_defectors(r_base)

        # # detached cells (neither in tumor nor touching it) outside baseline
        # detached_cells = self._find_detached_non_tumor_cells(Tumorcells, r_base, tips)
        # num_defectors = len(detached_cells)

        # # clusters of FC outside tumor
        # clustercells = []
        # cluster_compositions = self._collect_clusters_outside_tumor(Tumor, Tumorcells, clustercells)
        # clusters = len(cluster_compositions)

        # # stable IDs for clusters
        # if cluster_compositions:
            # cluster_compositions.sort(key=lambda c: (c["centroid_x"], c["centroid_y"], c["centroid_z"]))
            # for i, c in enumerate(cluster_compositions, start=1):
                # c["ClusterID_frame"] = i
            # self._assign_stable_ids(cluster_compositions)

        # curr_member_union = set().union(*(set(c["member_ids"]) for c in cluster_compositions)) if cluster_compositions else set()
        # singleton_ids = set(detached_cells) - curr_member_union

        # merges, splits, dissolves, curr_info = self._detect_merge_split(
            # self.prev_clusters_info, cluster_compositions,
            # curr_tumor_ids=set(Tumorcells),
            # singleton_ids=singleton_ids
        # )
        # for evt in merges:
            # self.events_writer.writerow([mcs, "merge", evt["parents"], [evt["child"]], evt["fractions"], "", ""])
        # for evt in splits:
            # self.events_writer.writerow([mcs, "split", [evt["parent"]], evt["children"], evt["fractions"], "", ""])
        # for evt in dissolves:
            # self.events_writer.writerow([mcs, "dissolve", [evt["parent"]], [], [],
                                         # round(evt["lost_to_singletons"],3),
                                         # round(evt["lost_to_main_tumor"],3)])
        # self.prev_clusters_info = curr_info

        # if mcs % 10 == 0:
            # self.save_cluster_compositions(cluster_compositions, mcs)

        # # spherical boundary scan over many directions
        # theta, phi, r_main, r_outer, r_base_arr = self._scan_boundaries_spherical(Tumorcells, clustercells, n_dirs=768, r_base=r_base)

        # # “invasive/infiltrative volumes” (solid-angle integral)
        # invasive_vol, infiltrative_vol = self._volumes_spherical(theta, phi, r_main, r_outer, r_base_arr)

        # # fingers from radial profile (use r_main vs direction index)
        # peaks, branches = find_peaks(r_main, prominence=5, distance=5, width=3)
        # branches = len(peaks)

        # # write per-step metrics
        # self.metrics_writer.writerow([mcs, invasive_vol, infiltrative_vol, branches,
                                      # len(defectorcells), num_defectors, clusters])

        # # snapshots
        # if mcs % 100 == 0 and len(theta):
            # boundary_rows = [[float(theta[i]), float(phi[i]), float(r_main[i]), float(r_outer[i]), float(r_base_arr[i])] for i in range(len(theta))]
            # def_pos = self._positions(detached_cells)
            # tumor_positions, tumor_leaders, tumor_followers = self._tumor_positions(Tumorcells)
            # self.save_mcs_data(self.output_dir1, mcs,
                               # boundary_rows, def_pos, tumor_positions, tumor_leaders, tumor_followers, spherical=True)

        # # final text log
        # if mcs == final_step - 1:
            # Surface = nx.subgraph(Tumor, surface)
            # perimeter = self._perimeter(Surface)
            # invarea_like = self._invasive_volume_minus_base(Tumor, clustercells, defectorcells)
            # complexity = (np.square(perimeter) / (4 * np.pi * invarea_like)) if invarea_like > 0 else 0.0

            # self._final_text_log(mcs, perimeter, complexity, Surface,
                                 # invasive_vol, infiltrative_vol,
                                 # len(defectorcells), num_defectors,
                                 # branches, clusters, cluster_compositions)
            # try:
                # self.events_file.close()
            # except Exception:
                # pass

    # # ---------- outputs ----------
    # def _init_outputs(self):
        # self.f = open(self.output_dir + f"/data3D_{Jlf}_{mu}_{PP}.txt", "a")
        # self.metrics_file = open(os.path.join(self.output_dir, f"Metrics3D_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        # self.metrics_writer = csv.writer(self.metrics_file)
        # self.metrics_writer.writerow(["MCS", "Invasive_Vol", "Infiltrative_Vol", "Fingers",
                                      # "Single_Defects", "Detached_Cells", "Clusters"])
        # self.output_dir1 = self.output_dir + f"/PositionData3D_{Jlf}_{mu}_{PP}"
        # os.makedirs(self.output_dir1, exist_ok=True)

        # self.events_file = open(os.path.join(self.output_dir, f"ClusterEvents3D_{Jlf}_{mu}_{PP}.csv"), "w", newline="")
        # self.events_writer = csv.writer(self.events_file)
        # self.events_writer.writerow(["MCS","Event","Parents","Children","OverlapFractions","LostToSingletons","LostToMainTumor"])

    # def _init_plots(self):
        # self.plot_win1 = self.add_new_plot_window(
            # title='3D Volumes Over Time', x_axis_title='MCS', y_axis_title='Volume',
            # x_scale_type='linear', y_scale_type='linear', grid=False, config_options={'legend': True}
        # )
        # self.plot_win1.add_plot("Invasive Vol", style='Lines', color='yellow', size=2)
        # self.plot_win1.add_plot("Infiltrative Vol", style='Lines', color='red', size=2)

    # def save_cluster_compositions(self, clusters_data, mcs):
        # mcs_folder = os.path.join(self.output_dir1, f"MCS_{mcs}")
        # os.makedirs(mcs_folder, exist_ok=True)
        # path = os.path.join(mcs_folder, f"ClusterComposition3D_{Jlf}_{mu}_{PP}_MCS_{mcs}.csv")
        # with open(path, "w", newline="") as f:
            # w = csv.writer(f)
            # w.writerow(["ClusterID_stable", "ClusterID_frame", "Leader Cells", "Follower Cells", "Total Cells",
                        # "Centroid_X", "Centroid_Y", "Centroid_Z",
                        # "Member_LC_XYZ", "Member_FC_XYZ"])
            # for c in clusters_data:
                # w.writerow([
                    # c.get("ClusterID_stable"), c.get("ClusterID_frame"),
                    # c["leader_cells"], c["follower_cells"], c["total_cells"],
                    # c["centroid_x"], c["centroid_y"], c["centroid_z"],
                    # str(c["leader_xyz"]), str(c["follower_xyz"])
                # ])

    # def save_mcs_data(self, output_dir1, mcs, boundary_data, def_pos, tumor_pos, tumor_L, tumor_F, spherical=True):
        # mcs_folder = os.path.join(output_dir1, f"MCS_{mcs}")
        # os.makedirs(mcs_folder, exist_ok=True)
        # tag = f"{Jlf}_{mu}_{PP}_MCS_{mcs}"

        # def save_csv(name, header, rows):
            # p = os.path.join(mcs_folder, f"{name}_{tag}.csv")
            # with open(p, "w", newline="") as fh:
                # w = csv.writer(fh)
                # w.writerow(header); w.writerows(rows)

        # if boundary_data:
            # if spherical:
                # save_csv("BoundaryData3D", ["Theta", "Phi", "R_Main", "R_Outer", "R_Base"], boundary_data)
            # else:
                # save_csv("BoundaryData", ["X","Y","Z","..."], boundary_data)

        # if def_pos:     save_csv("Defectors", ["id","x","y","z"], def_pos)
        # if tumor_pos:   save_csv("TumorXYZ", ["id","x","y","z"], tumor_pos)
        # if tumor_L:     save_csv("TumorLC",  ["id","x","y","z"], tumor_L)
        # if tumor_F:     save_csv("TumorFC",  ["id","x","y","z"], tumor_F)

    # # ---------- BFS / tumor core ----------
    # def _bfs(self, G, surface, start_cell):
        # visited, q = set([start_cell.id]), [start_cell.id]
        # G.add_node(start_cell.id, xCOM=start_cell.xCOM, yCOM=start_cell.yCOM, zCOM=start_cell.zCOM)
        # while q:
            # cid = q.pop(0)
            # c = self.fetch_cell_by_id(cid)
            # for neigh, area in self.get_cell_neighbor_data_list(c):
                # if neigh:
                    # if neigh.id not in visited:
                        # visited.add(neigh.id); q.append(neigh.id)
                        # G.add_node(neigh.id, xCOM=neigh.xCOM, yCOM=neigh.yCOM, zCOM=neigh.zCOM)
                    # dx = c.xCOM - neigh.xCOM; dy = c.yCOM - neigh.yCOM; dz = c.zCOM - neigh.zCOM
                    # G.add_edge(c.id, neigh.id, weight=(dx*dx+dy*dy+dz*dz)**0.5)
                # else:
                    # if c.id not in surface:
                        # surface.append(c.id)
        # return visited

    # def _collect_main_tumor(self, G, surface):
        # cx, cy, cz = self.dim.x//2, self.dim.y//2, self.dim.z//2
        # start_cell = self.cell_field[cx, cy, cz]
        # if not start_cell:
            # # expand until we hit any cell
            # for r in range(1, max(self.dim.x, self.dim.y, self.dim.z)):
                # for dx in (-r,0,r):
                    # for dy in (-r,0,r):
                        # for dz in (-r,0,r):
                            # x,y,z = cx+dx, cy+dy, cz+dz
                            # if 0<=x<self.dim.x and 0<=y<self.dim.y and 0<=z<self.dim.z:
                                # c = self.cell_field[x,y,z]
                                # if c: start_cell=c; break
                        # if start_cell: break
                    # if start_cell: break
                # if start_cell: break
        # if not start_cell: return set()
        # return self._bfs(G, surface, start_cell)

    # # ---------- radii helpers ----------
    # def _center_radius(self, cell):
        # cx, cy, cz = self.dim.x//2, self.dim.y//2, self.dim.z//2
        # return math.sqrt((cell.xCOM-cx)**2 + (cell.yCOM-cy)**2 + (cell.zCOM-cz)**2)

    # def _baseline_radius(self, Tumorcells):
        # vals = []
        # for cid in Tumorcells:
            # c = self.fetch_cell_by_id(cid)
            # if c: vals.append(self._center_radius(c))
        # return min(vals) if vals else 0.0

    # # ---------- defector / detached ----------
    # def _find_single_leader_defectors(self, r_base):
        # out = []
        # for cell in self.cell_list_by_type(self.LC):
            # n = sum(1 for neigh,_ in self.get_cell_neighbor_data_list(cell) if neigh)
            # if n == 0 and self._center_radius(cell) > r_base:
                # out.append(cell.id)
        # return out

    # def _find_detached_non_tumor_cells(self, Tumorcells, r_base, tips):
        # out, T = [], set(Tumorcells)
        # for cell in self.cell_list_by_type(self.LC, self.FC):
            # if cell.id in T: continue
            # neigh_ids = [n.id for n,_ in self.get_cell_neighbor_data_list(cell) if n]
            # if all(nid not in T for nid in neigh_ids):
                # if self._center_radius(cell) > r_base:
                    # out.append(cell.id); tips[cell] = 100
        # return out

    # # ---------- clusters outside tumor ----------
    # def _collect_clusters_outside_tumor(self, G, Tumorcells, clustercells):
        # T, checked = set(Tumorcells), set()
        # out = []
        # for cell in self.cell_list_by_type(self.FC):
            # if cell.id in T or cell.id in checked: 
                # continue
            # members = self._bfs(G, [], cell)
            # if len(members) >= 2:
                # checked.update(members); clustercells.extend(members)
                # leaders = 0; followers = 0
                # xs, ys, zs = [], [], []
                # leader_xyz, follower_xyz = [], []
                # for cid in members:
                    # c = self.fetch_cell_by_id(cid)
                    # if not c: continue
                    # xs.append(c.xCOM); ys.append(c.yCOM); zs.append(c.zCOM)
                    # if c.type == self.LC:
                        # leaders += 1; leader_xyz.append((c.xCOM,c.yCOM,c.zCOM))
                    # elif c.type == self.FC:
                        # followers += 1; follower_xyz.append((c.xCOM,c.yCOM,c.zCOM))
                # out.append({
                    # "leader_cells": leaders, "follower_cells": followers, "total_cells": len(members),
                    # "centroid_x": float(np.mean(xs)) if xs else 0.0,
                    # "centroid_y": float(np.mean(ys)) if ys else 0.0,
                    # "centroid_z": float(np.mean(zs)) if zs else 0.0,
                    # "leader_xyz": leader_xyz, "follower_xyz": follower_xyz,
                    # "member_ids": list(members)
                # })
        # return out

    # # ---------- spherical boundary scan ----------
    # def _scan_boundaries_spherical(self, Tumorcells, clustercells, n_dirs=768, r_base=0.0):
        # cx, cy, cz = self.dim.x//2, self.dim.y//2, self.dim.z//2
        # T, C = set(Tumorcells), set(clustercells)
        # # fibonacci sphere directions
        # dirs = []
        # for i in range(n_dirs):
            # z = 1 - 2*(i+0.5)/n_dirs
            # r = (1 - z*z) ** 0.5
            # phi = math.pi * (1 + 5**0.5) * i
            # x = r*math.cos(phi); y = r*math.sin(phi)
            # dirs.append((x,y,z))
        # r_main = np.zeros(n_dirs); r_outer = np.zeros(n_dirs)
        # theta = np.zeros(n_dirs); phi_arr = np.zeros(n_dirs)
        # rmax = math.sqrt(self.dim.x**2 + self.dim.y**2 + self.dim.z**2)

        # for i,(dx,dy,dz) in enumerate(dirs):
            # # spherical angles for bookkeeping
            # rr = (dx*dx+dy*dy+dz*dz)**0.5
            # th = math.atan2(dy, dx) % (2*math.pi)  # azimuth
            # ph = math.acos(dz/rr)                  # polar
            # theta[i] = th; phi_arr[i] = ph

            # step = 1.0; r = 0.0
            # lm = 0.0; lo = 0.0; has_m=False; has_o=False
            # while r < rmax:
                # x = int(round(cx + r*dx)); y = int(round(cy + r*dy)); z = int(round(cz + r*dz))
                # if not (0<=x<self.dim.x and 0<=y<self.dim.y and 0<=z<self.dim.z):
                    # break
                # c = self.cell_field[x,y,z]
                # if c:
                    # has_o = True; lo = r
                    # if (c.id in T) and (c.id not in C):
                        # has_m = True; lm = r
                # r += step
            # r_main[i] = lm if has_m else 0.0
            # r_outer[i] = lo if has_o else 0.0

        # r_base_arr = np.full(n_dirs, r_base, dtype=float)
        # return theta, phi_arr, r_main, r_outer, r_base_arr

    # def _volumes_spherical(self, theta, phi, r_main, r_outer, r_base):
        # # integrate over sphere using equal-area fibonacci samples: approximate
        # # weight per direction ~ 4π / N
        # N = len(theta)
        # if N == 0: return 0.0, 0.0
        # w = 4*math.pi / N
        # # volume between radii a and b is (4/3)π(b^3 - a^3); here per solid angle element ~ (1/3)(b^3 - a^3)
        # inv = np.sum((np.maximum(r_main, r_base)**3 - r_base**3) * (1.0/3.0) * w)
        # inf = np.sum((np.maximum(r_outer, r_base)**3 - r_base**3) * (1.0/3.0) * w)
        # return float(inv), float(inf)

    # # ---------- misc metrics ----------
    # def _positions(self, ids):
        # out=[]
        # for cid in ids:
            # c = self.fetch_cell_by_id(cid)
            # if c: out.append([cid, c.xCOM, c.yCOM, c.zCOM])
        # return out

    # def _tumor_positions(self, Tumorcells):
        # tumor, L, F = [], [], []
        # for cid in Tumorcells:
            # c = self.fetch_cell_by_id(cid)
            # if c:
                # row=[cid, c.xCOM, c.yCOM, c.zCOM]; tumor.append(row)
                # (L if c.type==self.LC else F if c.type==self.FC else tumor).append(row)
        # return tumor, L, F

    # def _invasive_volume_minus_base(self, Tumor, clustercells, defectorcells):
        # inv = 0.0; C=set(clustercells); D=set(defectorcells)
        # for nid in Tumor.nodes:
            # if nid in C or nid in D: continue
            # c = self.fetch_cell_by_id(nid)
            # if c: inv += c.volume
        # return inv

    # def _perimeter(self, Surface):
        # # graph edge-length sum in 3D
        # per = 0.0
        # for u,v,e in Surface.edges(data=True):
            # per += e.get('weight', 0.0)
        # return per

    # def _final_text_log(self, mcs, perimeter, complexity, Surface,
                        # invasive_vol, infiltrative_vol,
                        # single_defects, detached_cells_count,
                        # branches, clusters, cluster_compositions):
        # self.f.write(f"Step: {mcs}\n")
        # self.f.write(f"Tumor perimeter (graph sum): {perimeter}\n")
        # self.f.write(f"Tumor complexity: {complexity}\n")
        # self.f.write(f"Invasive Vol: {invasive_vol}\n")
        # self.f.write(f"Infiltrative Vol: {infiltrative_vol}\n")
        # self.f.write(f"Single Defects: {single_defects}\n")
        # self.f.write(f"Detached Cells: {detached_cells_count}\n")
        # self.f.write(f"Branches: {branches}\n")
        # self.f.write(f"Clusters: {clusters}\n")
        # self.f.write("Cluster Composition:\n")
        # for c in cluster_compositions:
            # self.f.write(
                # f"Cluster {c.get('ClusterID_stable')} (frame {c.get('ClusterID_frame')}) "
                # f"L:{c['leader_cells']} F:{c['follower_cells']} Tot:{c['total_cells']}\n"
            # )
        # self.f.write("END\n")
        # self.f.close(); self.metrics_file.close()

    # # ---------- stable IDs & events ----------
    # def _euclid(self, a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    # def _assign_stable_ids(self, comps):
        # if not comps: return
        # if not self.prev_tracks:
            # for c in sorted(comps, key=lambda c: (c["centroid_x"], c["centroid_y"], c["centroid_z"])):
                # cid = self.next_cluster_id; c["ClusterID_stable"]=cid
                # self.prev_tracks.append({"id":cid, "centroid":(float(c["centroid_x"]), float(c["centroid_y"]), float(c["centroid_z"]))})
                # self.next_cluster_id += 1
            # return
        # used=set(); new=[]
        # for c in comps:
            # cen=(float(c["centroid_x"]), float(c["centroid_y"]), float(c["centroid_z"]))
            # best=None; bestd=1e9
            # for tr in self.prev_tracks:
                # if tr["id"] in used: continue
                # d=self._euclid(cen, tr["centroid"])
                # if d<bestd: bestd=d; best=tr
            # if best and bestd<=self.centroid_match_radius:
                # c["ClusterID_stable"]=best["id"]; used.add(best["id"]); new.append({"id":best["id"], "centroid":cen})
            # else:
                # cid=self.next_cluster_id; c["ClusterID_stable"]=cid; new.append({"id":cid, "centroid":cen}); self.next_cluster_id+=1
        # self.prev_tracks=new

    # def _detect_merge_split(self, prev_info, curr_comps, curr_tumor_ids, singleton_ids,
                            # jaccard_min=0.15, cover_min=0.25):
        # curr_info=[{"id":c.get("ClusterID_stable"), "members":set(c.get("member_ids",[]))}
                   # for c in curr_comps if c.get("ClusterID_stable") is not None]

        # def jacc(a,b):
            # if not a or not b: return 0.0
            # inter=len(a&b); uni=len(a|b)
            # return 0.0 if uni==0 else inter/uni

        # merges=[]; splits=[]; dissolves=[]
        # # merges
        # for c in curr_info:
            # parents=[]; fr=[]
            # for p in prev_info:
                # J=jacc(p["members"], c["members"])
                # cov=len(p["members"] & c["members"]) / max(1, len(c["members"]))
                # if J>=jaccard_min and cov>=cover_min:
                    # parents.append(p["id"]); fr.append(round(cov,3))
            # if len(parents)>=2: merges.append({"child":c["id"], "parents":parents, "fractions":fr})
        # # splits & dissolves
        # for p in prev_info:
            # kids=[]; fr=[]
            # for c in curr_info:
                # J=jacc(p["members"], c["members"])
                # cov=len(p["members"] & c["members"]) / max(1, len(p["members"]))
                # if J>=jaccard_min and cov>=cover_min:
                    # kids.append(c["id"]); fr.append(round(cov,3))
            # if len(kids)>=2:
                # splits.append({"parent":p["id"], "children":kids, "fractions":fr})
            # elif len(kids)==0:
                # parent_sz = max(1, len(p["members"]))
                # dissolves.append({
                    # "parent": p["id"],
                    # "lost_to_singletons": len(p["members"] & singleton_ids)/parent_sz,
                    # "lost_to_main_tumor": len(p["members"] & curr_tumor_ids)/parent_sz
                # })
        # return merges, splits, dissolves, curr_info

