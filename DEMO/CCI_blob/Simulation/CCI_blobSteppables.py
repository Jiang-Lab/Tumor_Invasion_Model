# CCIecmSteppables.py
from cc3d.cpp.PlayerPython import *
from cc3d import CompuCellSetup
from cc3d.core.PySteppables import *

from datetime import datetime
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.signal import find_peaks
from numpy import trapz
import os
import csv
import math


# =========================
# Global knobs 
# =========================
k = 25      # (%) leaders target fraction in the blob
fgrow = 0.015   # follower growth increment per MCS 
lgrow = 0.010  

Jlf = 1         # Contact energy LCâ€“FC (overwritten into XML element id="J_LF")
mu  = 30        # Chemotaxis lambda for LC (overwritten into XML element id="lambda_chem")
PP  = 0.9       # Prob. a follower is proliferative (gets a division clock)
vol = 25

show_plots = 0  # 1 to render CC3D built-in plots; 0 for headless logging only


# ---- -------------------------------------------------------------------
CODE_TUMOR_CORE     = 5    # optional light tint for main mass
CODE_ENDPOINT       = 10   # stalk endpoints
CODE_SINGLETON      = 20   # detached single LCs
CODE_CLUSTER_CENTER = 30   # representative member closest to cluster centroid
# -----------------------------------------------------------------------------




# =========================================================
# ConstraintInitializerSteppable
# =========================================================
class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        super().__init__(frequency)
        self.cellcount_data = []

    def start(self):
        global k

        # count file
        self.cellcount_file = open(os.path.join(self.output_dir, f"CellCount_{Jlf}_{mu}_{PP}.csv"),
                                   "a", newline="")
        self.cellcount_writer = csv.writer(self.cellcount_file)
        self.cellcount_writer.writerow(["MCS", "Leader Cells", "Follower Cells", "Total"])

        # ensure k% LC inside the FC blob
        def frac_leaders():
            lc = len(self.cell_list_by_type(self.LC))
            total = lc + len(self.cell_list_by_type(self.FC))
            return 0.0 if total == 0 else lc / total

        # Randomly pick FCs in the blob and convert to LC until reaching k%
        while frac_leaders() < k / 100.0:
            # sample a random pixel within lattice; if FC, convert that pixel to a new LC cell
            x1 = np.random.randint(0, self.dim.x)
            y1 = np.random.randint(0, self.dim.y)
            c = self.cellField[x1, y1, 0]
            if c and c.type == self.FC:
                lc = self.new_cell(self.LC)
                self.cellField[x1, y1, 0] = lc

        
        for cell in self.cell_list_by_type(self.FC, self.LC):
            cell.targetVolume = vol
            cell.lambdaVolume = 2.0

        
        self.get_xml_element("J_LF").cdata = Jlf
        self.get_xml_element("lambda_chem").Lambda = mu

        # ----- RADIAL CHEMOTAXIS FIELD (MV = radial distance) -----
        mv = self.field.MV
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        g = 1.0  # scale: higher => weaker gradient values
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                r = math.hypot(x - cx, y - cy)
                mv[x, y, :] = r / g

    def step(self, mcs):
        final_step = self.simulator.getNumSteps()
        L = str(len(self.cell_list_by_type(self.LC)))
        F = str(len(self.cell_list_by_type(self.FC)))
        T = str(len(self.cell_list))
        self.cellcount_writer.writerow([mcs, L, F, T])

        if mcs == final_step - 1:
            self.cellcount_file.close()


# =========================================================
# GrowthSteppable 
# =========================================================
class GrowthSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        super().__init__(frequency)

    def step(self, mcs):
        for cell in self.cell_list_by_type(self.FC):
            if cell.targetVolume < 2 * vol:
                cell.targetVolume += fgrow


# =========================================================
# MitosisSteppable
# =========================================================
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        super().__init__(frequency)
        self.cell_to_proliferate = []

    def start(self):
        for cell in self.cell_list_by_type(self.FC):
            if np.random.rand() <= PP:
                cell.dict["clock"] = np.random.randint(0, 75)  # initial offset
            else:
                cell.dict["clock"] = None

    def step(self, mcs):
        cells_to_divide = []
        for cell in self.cell_list_by_type(self.FC):
            if cell.dict["clock"] is not None:
                cell.dict["clock"] += 1
                vary = np.random.randint(0, 50)
                if cell.volume > 2 * vol and cell.dict["clock"] > 75 + vary:
                    cells_to_divide.append(cell)

        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)

    def update_attributes(self):
        self.parent_cell.targetVolume /= 2.0
        self.parent_cell.dict["clock"] = 0
        self.clone_parent_2_child()
        # keep child same type as parent (followers divide into followers)
        if self.parent_cell.type == self.FC:
            self.child_cell.type = self.FC
        else:
            self.child_cell.type = self.LC


# =========================================================
# NeighborTrackerPrinterSteppable
# =========================================================
class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=10):
        super().__init__(frequency)

        # runtime buffers
        self.metrics_data = []
        self.boundary_data = []
        self.DefectorPosition_data = []
        self.TumorPosition_data = []

        # persistent cluster tracking
        self.prev_tracks = []
        self.next_cluster_id = 1
        self.centroid_match_radius = 12.0
        self.prev_clusters_info = []

    # ---------- lifecycle ----------
    def start(self):
        self._init_outputs()
        self._init_plots()
        self.create_scalar_field_cell_level_py("myField")
        
    
    def step(self, mcs):
        final_step = self.simulator.getNumSteps()

        # ---- cell-level scalar field (for visualization codes) ----
        tips = self.field.myField
        tips.clear()

        # ---- build tumor graph from center; find main mass & surface ----
        Tumor = nx.Graph()
        surface = []
        Tumorcells = self._collect_main_tumor(Tumor, surface)

        # baseline radius (closest tumor cell to center)
        baseline_r = self._baseline_radius(Tumorcells)

        # count LC inside tumor (stalk LC – just for reporting)
        stalklc, stalkcells = self._count_stalk_leaders(Tumorcells)

        # single LCs with no neighbors beyond baseline
        defectorcells = self._find_single_leader_defectors(baseline_r)

        # detached (neither in tumor nor touching it) beyond baseline; paint as you collect
        detached_cells = self._find_detached_non_tumor_cells(Tumorcells, baseline_r, tips)
        num_defectors = len(detached_cells)

        if show_plots:
            self.plot_win2.add_data_point('defectors', mcs, len(defectorcells))
            self.plot_win2.add_data_point('defectors+clusters', mcs, num_defectors)

        # clusters outside tumor; compositions + centroids
        clustercells = []
        cluster_compositions = self._collect_clusters_outside_tumor(Tumor, Tumorcells, clustercells)

        # ---- paint categories on myField ----
        # main tumor core
        for cid in Tumorcells:
            c = self.fetch_cell_by_id(cid)
            if c:
                tips[c] = CODE_TUMOR_CORE

        # endpoints on tumor surface (also pass tips so helper won’t error)
        Surface = nx.subgraph(Tumor, surface)
        endpoints, epnodes = self._surface_endpoints(Surface, tips)
        for c in endpoints:
            tips[c] = CODE_ENDPOINT

        # detached singletons
        for cid in detached_cells:
            c = self.fetch_cell_by_id(cid)
            if c:
                tips[c] = CODE_SINGLETON

        # cluster “centroid representatives”
        self._mark_cluster_centroids(tips, cluster_compositions, CODE_CLUSTER_CENTER)

        # ---- stable cluster IDs + merge/split/dissolve detection ----
        clusters = len(cluster_compositions)
        if cluster_compositions:
            cluster_compositions.sort(key=lambda c: (c["centroid_x"], c["centroid_y"]))
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

        # ---- polar boundary scan + areas/fingers ----
        theta, r_main, r_outer, r_base = self._scan_boundaries_polar(Tumorcells, clustercells, n_angles=360)
        invasive_area, infiltrative_area = self._areas_polar(theta, r_main, r_outer, r_base)
        finger_peaks, branches = self._fingers_polar(r_main)

        if show_plots:
            self.plot_win1.add_data_point('Invasive Area', mcs, invasive_area)
            self.plot_win1.add_data_point('Infiltrative Area', mcs, infiltrative_area)

        # ---- per-step metrics CSV ----
        self.metrics_writer.writerow([
            mcs, invasive_area, infiltrative_area, branches,
            len(defectorcells), num_defectors, clusters
        ])

        # ---- periodic CSV dumps of geometry/positions ----
        if mcs % 100 == 0 and len(theta):
            boundary_data = [[float(theta[j]), float(r_main[j]), float(r_outer[j]), float(r_base[j])]
                             for j in range(len(theta))]
            defector_positions = self._positions(detached_cells)
            tumor_positions, tumor_leaders, tumor_followers = self._tumor_positions(Tumorcells)
            self.save_mcs_data(self.output_dir1, mcs,
                               boundary_data,
                               defector_positions,
                               tumor_positions,
                               tumor_leaders,
                               tumor_followers,
                               polar=True)

        # ---- final snapshot + text log ----
        if mcs == final_step - 1:
            perimeter = self._perimeter(Surface)
            invarea = self._invasive_volume_minus_base(Tumor, clustercells, defectorcells)
            complexity = (np.square(perimeter) / (4 * np.pi * invarea)) if invarea > 0 else 0.0

            if show_plots and len(theta):
                xs_m, ys_m = self._polar_to_xy(theta, r_main)
                xs_o, ys_o = self._polar_to_xy(theta, r_outer)
                xs_b, ys_b = self._polar_to_xy(theta, r_base)
                for x, y in zip(xs_m, ys_m):
                    self.plot_win6.add_data_point("Tumor_Boundary", x, y)
                for x, y in zip(xs_o, ys_o):
                    self.plot_win6.add_data_point("Outer_Boundary", x, y)
                    self.plot_win6.add_data_point("Outer_Boundary_Curve", x, y)
                for x, y in zip(xs_b, ys_b):
                    self.plot_win6.add_data_point("Lowest Tumor Boundary Line", x, y)
                for p in finger_peaks:
                    xb, yb = xs_m[p], ys_m[p]
                    self.plot_win6.add_data_point("Branch Points", xb, yb)
                for cell_id in set(Tumorcells):
                    cell = self.fetch_cell_by_id(cell_id)
                    if cell:
                        self.plot_win6.add_data_point("Main Tumor Cells", cell.xCOM, cell.yCOM)
                        if cell.type == self.LC:
                            self.plot_win6.add_data_point("Tumor Leader Cells", cell.xCOM, cell.yCOM)
                        elif cell.type == self.FC:
                            self.plot_win6.add_data_point("Tumor Follower Cells", cell.xCOM, cell.yCOM)
                for cell_id in detached_cells:
                    cell = self.fetch_cell_by_id(cell_id)
                    if cell:
                        self.plot_win6.add_data_point("Defected Cells", cell.xCOM, cell.yCOM)

            # write final text summary
            min_height = 0.0  
            endpoints, epnodes = self._surface_endpoints(Surface, tips)  
            epheight, avgheight, heightvar, max_height = self._endpoint_heights_radius(epnodes)
            self._final_text_log(mcs, perimeter, complexity, endpoints,
                                 Tumor, invasive_area, infiltrative_area,
                                 len(defectorcells), num_defectors,
                                 branches, stalklc, avgheight, heightvar,
                                 max_height, min_height, clusters,
                                 cluster_compositions)


    # def step(self, mcs):
        # final_step = self.simulator.getNumSteps()
        # xmax, ymax = self.dim.x, self.dim.y

        # tips = self.field.myField
        # tips.clear()

        # Tumor = nx.Graph()
        # surface = []
        # Tumorcells = self._collect_main_tumor(Tumor, surface)

        # baseline_r = self._baseline_radius(Tumorcells)
        # # We count LC inside tumor (stalk LC)
        # stalklc, stalkcells = self._count_stalk_leaders(Tumorcells)

        # #SINGLE LEADER DEFECTORS (radial)
        # defectorcells = self._find_single_leader_defectors(baseline_r)

        # # detached cells outside main tumor and above baseline radius
        # detached_cells = self._find_detached_non_tumor_cells(Tumorcells, baseline_r, tips)
        # num_defectors = len(detached_cells)

        # if show_plots:
            # self.plot_win2.add_data_point('defectors', mcs, len(defectorcells))
            # self.plot_win2.add_data_point('defectors+clusters', mcs, num_defectors)

        # # CLUSTERS OUTSIDE TUMOR
        # clustercells = []
        # cluster_compositions = self._collect_clusters_outside_tumor(Tumor, Tumorcells, clustercells)
        
        # clusters = len(cluster_compositions)
        # if cluster_compositions:
            # cluster_compositions.sort(key=lambda c: (c["centroid_x"], c["centroid_y"]))
            # for i, c in enumerate(cluster_compositions, start=1):
                # c["ClusterID_frame"] = i
            # self._assign_stable_ids(cluster_compositions)

        # curr_member_union = set().union(*(set(c["member_ids"]) for c in cluster_compositions)) \
            # if cluster_compositions else set()
        # singleton_ids = set(detached_cells) - curr_member_union

        # merges, splits, dissolves, curr_info = self._detect_merge_split(
            # self.prev_clusters_info,
            # cluster_compositions,
            # curr_tumor_ids=set(Tumorcells),
            # singleton_ids=singleton_ids
        # )
        # for evt in merges:
            # self.events_writer.writerow([mcs, "merge", evt["parents"], [evt["child"]], evt["fractions"], "", ""])
        # for evt in splits:
            # self.events_writer.writerow([mcs, "split", [evt["parent"]], evt["children"], evt["fractions"], "", ""])
        # for evt in dissolves:
            # self.events_writer.writerow([
                # mcs, "dissolve", [evt["parent"]], [], [],
                # round(evt["lost_to_singletons"], 3),
                # round(evt["lost_to_main_tumor"], 3)
            # ])
        # self.prev_clusters_info = curr_info

        # if mcs % 10 == 0:
            # self.save_cluster_compositions(cluster_compositions, mcs)

        # # POLAR BOUNDARIES
        # theta, r_main, r_outer, r_base = self._scan_boundaries_polar(Tumorcells, clustercells, n_angles=360)

        # # AREAS / FINGERS (POLAR)
        # invasive_area, infiltrative_area = self._areas_polar(theta, r_main, r_outer, r_base)
        # finger_peaks, branches = self._fingers_polar(r_main)
        # if show_plots:
            # self.plot_win1.add_data_point('Invasive Area', mcs, invasive_area)
            # self.plot_win1.add_data_point('Infiltrative Area', mcs, infiltrative_area)

        # # write metrics
        # self.metrics_writer.writerow([mcs, invasive_area, infiltrative_area, branches,
                                      # len(defectorcells), num_defectors, clusters])

        # # GRAPH-BASED STALKS / ENDPOINTS / PERIMETER/COMPLEXITY
        # Surface = nx.subgraph(Tumor, surface)
        # endpoints, epnodes = self._surface_endpoints(Surface, tips)
        # stalks = len(endpoints)
        # min_height = 0.0  # no longer used for vertical base; keep for text log compatibility
        # epheight, avgheight, heightvar, max_height = self._endpoint_heights_radius(epnodes)
        # invarea = self._invasive_volume_minus_base(Tumor, clustercells, defectorcells)

        # # PERIODIC CSV OUTPUTS (polar boundary arrays)
        # if mcs % 100 == 0 and len(theta):
            # boundary_data = [[float(theta[j]), float(r_main[j]), float(r_outer[j]), float(r_base[j])]
                             # for j in range(len(theta))]
            # defector_positions = self._positions(detached_cells)
            # tumor_positions, tumor_leaders, tumor_followers = self._tumor_positions(Tumorcells)
            # self.save_mcs_data(self.output_dir1, mcs,
                               # boundary_data,
                               # defector_positions,
                               # tumor_positions,
                               # tumor_leaders,
                               # tumor_followers,
                               # polar=True)

        # # FINAL SNAPSHOT
        # if mcs == final_step - 1:
            # perimeter = self._perimeter(Surface)
            # complexity = (np.square(perimeter) / (4 * np.pi * invarea)) if invarea > 0 else 0.0

            # #XY preview of polar boundaries
            # if show_plots and len(theta):
                # xs_m, ys_m = self._polar_to_xy(theta, r_main)
                # xs_o, ys_o = self._polar_to_xy(theta, r_outer)
                # xs_b, ys_b = self._polar_to_xy(theta, r_base)

                # for x, y in zip(xs_m, ys_m):
                    # self.plot_win6.add_data_point("Tumor_Boundary", x, y)
                # for x, y in zip(xs_o, ys_o):
                    # self.plot_win6.add_data_point("Outer_Boundary", x, y)
                    # self.plot_win6.add_data_point("Outer_Boundary_Curve", x, y)
                # for x, y in zip(xs_b, ys_b):
                    # self.plot_win6.add_data_point("Lowest Tumor Boundary Line", x, y)

                # for p in finger_peaks:
                    # xb, yb = xs_m[p], ys_m[p]
                    # self.plot_win6.add_data_point("Branch Points", xb, yb)

                # # overlay cells
                # for cell_id in set(Tumorcells):
                    # cell = self.fetch_cell_by_id(cell_id)
                    # if cell:
                        # self.plot_win6.add_data_point("Main Tumor Cells", cell.xCOM, cell.yCOM)
                        # if cell.type == self.LC:
                            # self.plot_win6.add_data_point("Tumor Leader Cells", cell.xCOM, cell.yCOM)
                        # elif cell.type == self.FC:
                            # self.plot_win6.add_data_point("Tumor Follower Cells", cell.xCOM, cell.yCOM)
                # for cell_id in detached_cells:
                    # cell = self.fetch_cell_by_id(cell_id)
                    # if cell:
                        # self.plot_win6.add_data_point("Defected Cells", cell.xCOM, cell.yCOM)

            # # text log
            # self._final_text_log(mcs, perimeter, complexity, endpoints,
                                 # Tumor, invasive_area, infiltrative_area,
                                 # len(defectorcells), num_defectors,
                                 # branches, stalklc, avgheight, heightvar,
                                 # max_height, min_height, clusters,
                                 # cluster_compositions)

    def finish(self):
        if self.output_dir is not None and show_plots:
            png_output_path6 = Path(self.output_dir).joinpath(f"MainTumor_{Jlf}_{mu}_{PP}.png")
            self.plot_win6.save_plot_as_png(png_output_path6, 1000, 1000)
        try:
            self.events_file.close()
        except Exception:
            pass

    # ---------- outputs / plots ----------
    def _init_outputs(self):
        now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        self.f = open(self.output_dir + f"/data_{Jlf}_{mu}_{PP}.txt", "a")

        self.metrics_file = open(os.path.join(self.output_dir, f"Metrics_Data_{Jlf}_{mu}_{PP}.csv"),
                                 "a", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(["MCS", "Invasive Area", "Infiltrative Area", "Fingers",
                                      "Single Defects", "Detached Cells", "Clusters"])

        self.output_dir1 = self.output_dir + f"/PositionData_{Jlf}_{mu}_{PP}"
        os.makedirs(self.output_dir1, exist_ok=True)

        # events CSV
        self.events_file = open(os.path.join(self.output_dir, f"ClusterEvents_{Jlf}_{mu}_{PP}.csv"),
                                "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(
            ["MCS", "Event", "Parents", "Children", "OverlapFractions",
             "LostToSingletons", "LostToMainTumor"]
        )

    def _init_plots(self):
        if not show_plots:
            return
        self.plot_win1 = self.add_new_plot_window(
            title='Areas Over Time: ' + f"{Jlf}_{mu}_{PP}",
            x_axis_title='MCS', y_axis_title='Area (lattice units)',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True}
        )
        self.plot_win2 = self.add_new_plot_window(
            title='Defectors Over Time: ' + f"{Jlf}_{mu}_{PP}",
            x_axis_title='MCS', y_axis_title='Number of Defectors',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True}
        )
        self.plot_win6 = self.add_new_plot_window(
            title='Polar Boundary Preview (XY)',
            x_axis_title='X', y_axis_title='Y',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True}
        )
        self.plot_win1.add_plot("Invasive Area", style='Lines', color='yellow', size=2)
        self.plot_win1.add_plot("Infiltrative Area", style='Lines', color='red', size=2)
        self.plot_win2.add_plot("defectors", style='Lines', color='green', size=5)
        self.plot_win2.add_plot("defectors+clusters", style='Lines', color='blue', size=2)
        self.plot_win6.add_plot("Tumor_Boundary", style='Lines', color='yellow', size=2)
        self.plot_win6.add_plot("Lowest Tumor Boundary Line", style='Lines', color='purple', size=5)
        self.plot_win6.add_plot("Branch Points", style='Dots', color='purple', size=5)
        self.plot_win6.add_plot("Defected Cells", style="Dots", color="blue", size=5)
        self.plot_win6.add_plot("Outer_Boundary", style='Dots', color='orange', size=5)
        self.plot_win6.add_plot("Outer_Boundary_Curve", style='Lines', color='red', size=2)
        self.plot_win6.add_plot("Main Tumor Cells", style='Dots', color='Green', size=5)
        self.plot_win6.add_plot("Tumor Leader Cells", style='Dots', color='lime', size=6)
        self.plot_win6.add_plot("Tumor Follower Cells", style='Dots', color='cyan', size=6)
        self.plot_win6.add_plot("Cluster Centroids", style='Star', color='orange', size=6)

    def save_cluster_compositions(self, clusters_data, mcs):
        mcs_folder = os.path.join(self.output_dir1, f"MCS_{mcs}")
        os.makedirs(mcs_folder, exist_ok=True)
        cluster_filename = os.path.join(mcs_folder, f"ClusterComposition_{Jlf}_{mu}_{PP}_MCS_{mcs}.csv")
        with open(cluster_filename, "w", newline="") as cluster_file:
            cluster_writer = csv.writer(cluster_file)
            cluster_writer.writerow([
                "Cluster ID", "ClusterID_frame",
                "Leader Cells", "Follower Cells", "Total Cells",
                "Centroid_X", "Centroid_Y",
                "Member_Leader_X", "Member_Leader_Y",
                "Member_Follower_X", "Member_Follower_Y"
            ])
            for cluster in clusters_data:
                cluster_writer.writerow([
                    cluster.get("ClusterID_stable"),
                    cluster.get("ClusterID_frame"),
                    cluster["leader_cells"], cluster["follower_cells"], cluster["total_cells"],
                    cluster["centroid_x"], cluster["centroid_y"],
                    str(cluster["leader_xs"]), str(cluster["leader_ys"]),
                    str(cluster["follower_xs"]), str(cluster["follower_ys"])
                ])

    def save_mcs_data(self, output_dir1, mcs,
                      boundary_data,
                      defector_positions,
                      tumor_positions,
                      tumor_leader_cells,
                      tumor_follower_cells,
                      polar=True):
        mcs_folder = os.path.join(output_dir1, f"MCS_{mcs}")
        os.makedirs(mcs_folder, exist_ok=True)
        param_tag = f"{Jlf}_{mu}_{PP}_MCS_{mcs}"

        def save_csv(filename_prefix, header, data):
            filename = f"{filename_prefix}_{param_tag}.csv"
            path = os.path.join(mcs_folder, filename)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)

        if boundary_data:
            if polar:
                save_csv("BoundaryData", ["Theta", "R_Main", "R_Outer", "R_Base"], boundary_data)
            else:
                save_csv("BoundaryData", ["X", "Main_Tumor", "Outermost", "Lowest_boundary_point"], boundary_data)

        if defector_positions:
            save_csv("DefectorPosition", ["id", "x", "y"], defector_positions)
        if tumor_positions:
            save_csv("TumorPosition", ["id", "x", "y"], tumor_positions)
        if tumor_leader_cells:
            save_csv("TumorLeaderCells", ["id", "x", "y"], tumor_leader_cells)
        if tumor_follower_cells:
            save_csv("TumorFollowerCells", ["id", "x", "y"], tumor_follower_cells)

    # ---------- BFS / graph ops ----------
    def _bfs(self, Tumor, surface, start_cell):
        visited = set()
        queue = []
        visited.add(start_cell.id)
        queue.append(start_cell.id)
        Tumor.add_node(start_cell.id, xCOM=start_cell.xCOM, yCOM=start_cell.yCOM)
        while queue:
            mID = queue.pop(0)
            m = self.fetch_cell_by_id(mID)
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(m):
                if neighbor:
                    x = neighbor.xCOM
                    y = neighbor.yCOM
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        queue.append(neighbor.id)
                        Tumor.add_node(neighbor.id, xCOM=x, yCOM=y)
                    w = math.hypot(m.xCOM - x, m.yCOM - y)
                    Tumor.add_edge(m.id, neighbor.id, weight=w)
                if not neighbor:
                    if m.id not in surface:
                        surface.append(m.id)
        return visited

    # ---------- main tumor / baseline / defectors ----------
    def _collect_main_tumor(self, Tumor, surface):
        """BFS from the center blob."""
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        start_cell = self.cell_field[cx, cy, 0]

        if not start_cell:
            # expand search rings until a cell is found
            max_r = max(self.dim.x, self.dim.y)
            found = None
            for r in range(1, max_r):
                # top/bottom
                for dx in range(-r, r + 1):
                    for dy in (-r, r):
                        x1, y1 = cx + dx, cy + dy
                        if 0 <= x1 < self.dim.x and 0 <= y1 < self.dim.y:
                            c = self.cell_field[x1, y1, 0]
                            if c:
                                found = c
                                break
                    if found:
                        break
                if not found:
                    # left/right
                    for dy in range(-r + 1, r):
                        for dx in (-r, r):
                            x1, y1 = cx + dx, cy + dy
                            if 0 <= x1 < self.dim.x and 0 <= y1 < self.dim.y:
                                c = self.cell_field[x1, y1, 0]
                                if c:
                                    found = c
                                    break
                        if found:
                            break
                if found:
                    start_cell = found
                    break

        if not start_cell:
            return set()

        return self._bfs(Tumor, surface, start_cell)

    def _center_radius(self, cell):
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        return math.hypot(cell.xCOM - cx, cell.yCOM - cy)

    def _baseline_radius(self, Tumorcells):
        if not Tumorcells:
            return 0.0
        vals = []
        for cid in Tumorcells:
            c = self.fetch_cell_by_id(cid)
            if c:
                vals.append(self._center_radius(c))
        return min(vals) if vals else 0.0

    def _count_stalk_leaders(self, Tumorcells):
        stalklc = 0
        stalkcells = []
        for cell in self.cell_list_by_type(self.LC):
            if cell.id in Tumorcells:
                stalklc += 1
                stalkcells.append(cell)
        return stalklc, stalkcells

    def _find_single_leader_defectors(self, baseline_r):
        defectorcells = []
        for cell in self.cell_list_by_type(self.LC):
            n = 0
            for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                if neighbor:
                    n += 1
            if n == 0 and self._center_radius(cell) > baseline_r:
                defectorcells.append(cell.id)
        return defectorcells

    # def _find_detached_non_tumor_cells(self, Tumorcells, baseline_r, tips):
        # detached = []
        # Tumorcells_set = set(Tumorcells)
        # for cell in self.cell_list_by_type(self.LC, self.FC):
            # if cell.id in Tumorcells_set:
                # continue
            # neighbors = [neigh for neigh, _ in self.get_cell_neighbor_data_list(cell) if neigh]
            # if all(neigh.id not in Tumorcells_set for neigh in neighbors):
                # if self._center_radius(cell) > baseline_r:
                    # detached.append(cell.id)
                    # tips[cell] = 100
        # return detached
        
    def _find_detached_non_tumor_cells(self, Tumorcells, baseline_r, tips):
        detached = []
        Tumorcells_set = set(Tumorcells)
        for cell in self.cell_list_by_type(self.LC, self.FC):
            if cell.id in Tumorcells_set:
                continue
            neighbors = [neigh for neigh, _ in self.get_cell_neighbor_data_list(cell) if neigh]
            if all(neigh.id not in Tumorcells_set for neigh in neighbors):
                if self._center_radius(cell) > baseline_r:
                    detached.append(cell.id)
                    if tips is not None:
                        tips[cell] = CODE_SINGLETON   
        return detached
    
        

    # ---------- clusters outside tumor ----------
    def _collect_clusters_outside_tumor(self, Tumor, Tumorcells, clustercells):
        cluster_checked = set()
        cluster_compositions = []

        for cell in self.cell_list_by_type(self.FC):
            if cell.id in Tumorcells or cell.id in cluster_checked:
                continue
            new_cluster_cells = self._bfs(Tumor, [], cell)
            if len(new_cluster_cells) >= 2:
                cluster_checked.update(new_cluster_cells)
                clustercells.extend(new_cluster_cells)

                # counts
                num_leader_cells = sum(
                    1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.LC
                )
                num_follower_cells = sum(
                    1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.FC
                )

                # centroid
                xs = [self.fetch_cell_by_id(cid).xCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                ys = [self.fetch_cell_by_id(cid).yCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                centroid_x = float(np.mean(xs)) if xs else 0.0
                centroid_y = float(np.mean(ys)) if ys else 0.0

                leader_xs, leader_ys, follower_xs, follower_ys = [], [], [], []
                for cid in new_cluster_cells:
                    c = self.fetch_cell_by_id(cid)
                    if not c:
                        continue
                    if c.type == self.LC:
                        leader_xs.append(c.xCOM); leader_ys.append(c.yCOM)
                    elif c.type == self.FC:
                        follower_xs.append(c.xCOM); follower_ys.append(c.yCOM)

                cluster_compositions.append({
                    "leader_cells": num_leader_cells,
                    "follower_cells": num_follower_cells,
                    "total_cells": len(new_cluster_cells),
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "leader_xs": leader_xs,
                    "leader_ys": leader_ys,
                    "follower_xs": follower_xs,
                    "follower_ys": follower_ys,
                    "member_ids": list(new_cluster_cells)
                })

                if show_plots:
                    self.plot_win6.add_data_point("Cluster Centroids", centroid_x, centroid_y)

        return cluster_compositions

    # ---------- polar boundaries ----------
    def _scan_boundaries_polar(self, Tumorcells, clustercells, n_angles=180):
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        Tumorcells_set = set(Tumorcells)
        clustercells_set = set(clustercells)

        theta_vals = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
        r_main = np.zeros_like(theta_vals, dtype=float)
        r_outer = np.zeros_like(theta_vals, dtype=float)

        rmax = math.hypot(self.dim.x, self.dim.y)
        for ti, th in enumerate(theta_vals):
            r = 0.0
            step = 1.0
            last_main_r = 0.0
            last_outer_r = 0.0
            has_main = False
            has_outer = False

            while r < rmax:
                x = int(round(cx + r * math.cos(th)))
                y = int(round(cy + r * math.sin(th)))
                if 0 <= x < self.dim.x and 0 <= y < self.dim.y:
                    cell = self.cell_field[x, y, 0]
                    if cell:
                        has_outer = True
                        last_outer_r = r
                        if (cell.id in Tumorcells_set) and (cell.id not in clustercells_set):
                            has_main = True
                            last_main_r = r
                else:
                    break
                r += step

            r_main[ti] = last_main_r if has_main else 0.0
            r_outer[ti] = last_outer_r if has_outer else 0.0

        r_base = np.zeros_like(theta_vals)
        pos = r_main[r_main > 0]
        if len(pos):
            r_base[:] = float(np.min(pos))
        return theta_vals, r_main, r_outer, r_base

    def _areas_polar(self, theta, r_main, r_outer, r_base):
        rm2 = np.maximum(r_main, r_base) ** 2 - r_base ** 2
        ro2 = np.maximum(r_outer, r_base) ** 2 - r_base ** 2
        invasive = np.trapz(0.5 * rm2, theta)
        infiltrative = np.trapz(0.5 * ro2, theta)
        return invasive, infiltrative

    def _fingers_polar(self, r_main, prominence=5, distance=5, width=3):
        peaks, _ = find_peaks(r_main, prominence=prominence, distance=distance, width=width)
        return peaks, len(peaks)

    def _polar_to_xy(self, theta, r):
        cx = self.dim.x // 2
        cy = self.dim.y // 2
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        return xs, ys

    # ---------- stalk endpoints / perimeter / invarea ----------
    # def _surface_endpoints(self, Surface, tips):
        # endpoints, epnodes = [], []
        # for node in list(Surface.nodes):
            # if Surface.degree(node) <= 2:
                # cell = self.fetch_cell_by_id(node)
                # if cell:
                    # endpoints.append(cell)
                    # epnodes.append(node)
                    # tips[cell] = 100
        # return endpoints, epnodes
    
    
    def _surface_endpoints(self, Surface, tips):
        endpoints, epnodes = [], []
        for node in list(Surface.nodes):
            if Surface.degree(node) <= 2:
                cell = self.fetch_cell_by_id(node)
                if cell:
                    endpoints.append(cell)
                    epnodes.append(node)
                    if tips is not None:
                        tips[cell] = CODE_ENDPOINT    
        return endpoints, epnodes

    def _endpoint_heights_radius(self, epnodes):
        """Return radial 'heights' (distance from center) stats of endpoints."""
        vals = []
        for node in epnodes:
            c = self.fetch_cell_by_id(node)
            if c:
                vals.append(self._center_radius(c))
        if len(vals) == 0:
            return [], 0.0, 0.0, 0.0
        vals = np.asarray(vals, dtype=float)
        return vals.tolist(), float(np.mean(vals)), float(np.var(vals)), float(np.max(vals))

    def _invasive_volume_minus_base(self, Tumor, clustercells, defectorcells):
        inv = 0
        clustercells = set(clustercells)
        defectorcells = set(defectorcells)
        for id_ in Tumor.nodes:
            cell = self.fetch_cell_by_id(id_)
            if cell and (id_ not in clustercells) and (id_ not in defectorcells):
                inv += cell.volume
        return inv

    def _perimeter(self, Surface):
        per = 0.0
        for u, v, e in Surface.edges(data=True):
            per += e.get('weight', 0.0)
        return per

    # ---------- periodic helpers ----------
    def _positions(self, ids):
        out = []
        for cid in ids:
            cell = self.fetch_cell_by_id(cid)
            if cell:
                out.append([cell.id, cell.xCOM, cell.yCOM])
        return out

    def _tumor_positions(self, Tumorcells):
        tumor_positions, tumor_leader_cells, tumor_follower_cells = [], [], []
        for cell_id in Tumorcells:
            cell = self.fetch_cell_by_id(cell_id)
            if cell:
                tumor_positions.append([cell.id, cell.xCOM, cell.yCOM])
                if cell.type == self.LC:
                    tumor_leader_cells.append([cell.id, cell.xCOM, cell.yCOM])
                elif cell.type == self.FC:
                    tumor_follower_cells.append([cell.id, cell.xCOM, cell.yCOM])
        return tumor_positions, tumor_leader_cells, tumor_follower_cells
    
    
    def _mark_cluster_centroids(self, tips, cluster_compositions, code_value):
        for comp in cluster_compositions:
            cx = float(comp.get("centroid_x", 0.0))
            cy = float(comp.get("centroid_y", 0.0))
            has_z = "centroid_z" in comp
            cz = float(comp.get("centroid_z", 0.0)) if has_z else 0.0

            best_cell = None
            best_d2 = float("inf")
            for cid in comp.get("member_ids", []):
                cell = self.fetch_cell_by_id(cid)
                if not cell:
                    continue
                dx = cell.xCOM - cx
                dy = cell.yCOM - cy
                if has_z:
                    dz = cell.zCOM - cz
                    d2 = dx*dx + dy*dy + dz*dz
                else:
                    d2 = dx*dx + dy*dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_cell = cell

            if best_cell is not None:
                tips[best_cell] = code_value
    
    
    # ---------- final outputs ----------
    def _final_text_log(self, mcs, perimeter, complexity, endpoints,
                        Tumor, invasive_area, infiltrative_area,
                        single_defects, detached_cells_count,
                        branches, stalklc, avgheight, heightvar,
                        max_height, min_height, clusters,
                        cluster_compositions):
        self.f.write("Step: " + str(mcs) + "\n")
        self.f.write("Tumor perimeter: " + str(perimeter) + "\n")
        self.f.write("Tumor complexity: " + str(complexity) + "\n")
        self.f.write("Tumor endpoints(stalks): " + str(len(endpoints)) + "\n")
        self.f.write("Tumor cells: " + str(len(list(Tumor.nodes))) + "\n")
        self.f.write("Invasive Area:" + str(invasive_area) + "\n")
        self.f.write("Infiltrative Area:" + str(infiltrative_area) + "\n")
        self.f.write("Single Defects: " + str(single_defects) + "\n")
        self.f.write("Detached Cells: " + str(detached_cells_count) + "\n")
        self.f.write("Branches: " + str(branches) + "\n")
        self.f.write("Stalk LC: " + str(stalklc) + "\n")
        self.f.write("Average Radial Height of Stalks: " + str(avgheight) + "\n")
        self.f.write("Variance in Radial Height of Stalks: " + str(heightvar) + "\n")
        self.f.write("max_radial_height: " + str(max_height) + "\n")
        self.f.write("min_height_legacy: " + str(min_height) + "\n")
        self.f.write("CLUSTER DATA:\n Total Clusters: " + str(clusters) + "\n")
        self.f.write("Cluster Composition:\n")
        for cluster in cluster_compositions:
            self.f.write(
                f"Cluster {cluster.get('ClusterID_stable')} "
                f"(frame {cluster.get('ClusterID_frame')}) - "
                f"Leader Cells: {cluster['leader_cells']}, "
                f"Follower Cells: {cluster['follower_cells']}, "
                f"Total Cells: {cluster['total_cells']}\n"
            )
        self.f.write("END")
        self.f.close()
        self.metrics_file.close()

    # ---------- stable ID tracker & merge/split ----------
    def _euclid(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _assign_stable_ids(self, comps):
        if not comps:
            return
        if not self.prev_tracks:
            for c in sorted(comps, key=lambda c: (c["centroid_x"], c["centroid_y"])):
                c["ClusterID_stable"] = self.next_cluster_id
                self.prev_tracks.append({
                    "id": self.next_cluster_id,
                    "centroid": (float(c["centroid_x"]), float(c["centroid_y"]))
                })
                self.next_cluster_id += 1
            return

        used_prev = set()
        new_tracks = []
        for c in comps:
            cen = (float(c["centroid_x"]), float(c["centroid_y"]))
            best = None
            best_d = float("inf")
            for tr in self.prev_tracks:
                if tr["id"] in used_prev:
                    continue
                d = self._euclid(cen, tr["centroid"])
                if d < best_d:
                    best_d = d
                    best = tr
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
            if not a or not b:
                return 0.0
            inter = len(a & b)
            uni = len(a | b)
            return 0.0 if uni == 0 else inter / uni

        merges, splits, dissolves = [], [], []

        # merges
        for c in curr_info:
            parents, fracs = [], []
            for p in prev_info:
                J = jacc(p["members"], c["members"])
                cover_child = len(p["members"] & c["members"]) / max(1, len(c["members"]))
                if J >= jaccard_min and cover_child >= cover_min:
                    parents.append(p["id"])
                    fracs.append(round(cover_child, 3))
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
