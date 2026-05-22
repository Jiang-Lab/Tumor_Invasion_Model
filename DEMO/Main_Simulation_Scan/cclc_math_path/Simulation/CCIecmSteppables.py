from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup

from cc3d.core.PySteppables import *
from datetime import datetime
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.signal import find_peaks
from numpy import trapz
import os
import csv
from itertools import product

k = 25 #(25) <1-10,10-80> This is the percent concentration of Leaders in the tumor
matrix = np.zeros((300,500), int)
lccelldiv=0
fccelldiv=0
divtime=0
fgrow=0.015 #(.11) <.005 to .015> growth rate for Followers 
lgrow = .010 #(.015) growth rate for Leaders (less than Followers)


Jlf = 2 ##The Adhesion (contact energy) of leader cells and follower cells
mu = 24  ##Chemotaxis Lambda
PP = 0.5 ## the percentage of the follower cells allowed to proliferate

show_plots = 0  # Set to 1 to show plots, or 0 to disable plotting


# iteration = {{iteration}}

Jlf = float(os.environ['JLF'])
mu  = float(os.environ['MU'])
PP  = float(os.environ['PP'])
rep = int(os.environ.get('REP', '0'))
iteration = int(os.environ.get('ITERATION', '-1'))



# iteration = int(os.environ.get("ITERATION", "0"))  ##Default to 0 if not set
# parameter_values = {
    # 'Jlf': [ -5,-4,-3,-2,-1,0,1,2,3,4,5],
    # 'mu': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
    # 'PP': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# }
# parameter_combinations = list(product(parameter_values['Jlf'], parameter_values['mu'], parameter_values['PP']))
# index = iteration % len(parameter_combinations) 
# if 0 <= index < len(parameter_combinations):
    # Jlf, mu, PP = parameter_combinations[index]




class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency=1): 
        SteppableBasePy.__init__(self,frequency)
        self.cellcount_data = []
    
    def start(self):
        global tracklc, k
        
        self.cellcount_file = open(os.path.join(self.output_dir, f"CellCount_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.cellcount_writer = csv.writer(self.cellcount_file)
        self.cellcount_writer.writerow(["MCS", "Leader Cells", "Follower Cells", "Total"])


        i= len(self.cell_list_by_type(self.LC))/(len(self.cell_list_by_type(self.LC))+len(self.cell_list_by_type(self.FC)))
        
        #This is for creating k% concentration of leader cells
        while i < k/100:
            lc = self.new_cell(self.LC)
            rand = np.random
            x1 = np.random.randint(1,self.dim.x)
            y1 = np.random.randint(1,20)
            c1 = self.cellField[x1,y1,0]
            if c1.type == 2:
                self.cellField[x1,y1,0] = lc 
                i= len(self.cell_list_by_type(self.LC))/(len(self.cell_list_by_type(self.LC))+len(self.cell_list_by_type(self.FC)))

   
        mv = self.field.MV
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                #mv[x, i, :] = x+i
                g=1 # (1) <0.1 to 15.0> raise this number to weaken the gradient field, or lower g to strengthen it
                mv[x, y, :] = y/g  
            

        for cell in self.cell_list_by_type(self.FC, self.LC):
            cell.targetVolume = 10
            cell.lambdaVolume = 2.0

        self.get_xml_element("J_LF").cdata = Jlf
        self.get_xml_element("lambda_chem").Lambda = mu

        
    def step(self, mcs): 
        final_step = self.simulator.getNumSteps()
        
        Leaders = str(len(self.cell_list_by_type(self.LC)))
        Followers = str(len(self.cell_list_by_type(self.FC)))
        Total = str(len(self.cell_list))
        
        self.cellcount_writer.writerow([mcs, Leaders, Followers, Total])
 
        if mcs == final_step-1:
            self.cellcount_file.close()
            
            
class GrowthSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self, frequency)
       
    def step(self, mcs):
        
        for cell in self.cell_list_by_type(self.FC):
            if cell.targetVolume<20:
                cell.targetVolume += fgrow 
                
                

class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.cell_to_proliferate = []

    def start(self):
        '''
        Initialize the clock for PP% of the Follower Cells (FC).
        If the cell is selected to proliferate (PP% proliferative probability), assign it a clock.
        If not, set the clock to None.
        '''
        for cell in self.cell_list_by_type(self.FC):
            if np.random.rand() <= PP:  # %percentage(proliferative probability)
                cell.dict["clock"] = np.random.randint(0, 75)  # Assign a random clock between 0 and 75
            else:
                cell.dict["clock"] = None  

    def step(self, mcs):
        cells_to_divide = []
        global lccelldiv, fccelldiv
        
        for cell in self.cell_list_by_type(self.FC):
            
            if cell.dict["clock"] is not None:
                cell.dict["clock"] += 1 
                vary = np.random.randint(0, 50)  

                if cell.volume > 20 and cell.dict["clock"] > 75 + vary:
                    cells_to_divide.append(cell)  # Add cell to the list for division

        # Perform the division for the selected cells
        for cell in cells_to_divide:
            self.divide_cell_random_orientation(cell)
            if cell.type == 2:
                fccelldiv += 1

    def update_attributes(self):
        self.parent_cell.targetVolume /= 2.0
        self.parent_cell.dict["clock"] = 0  
        self.clone_parent_2_child()

        if self.parent_cell.type == self.FC:
            self.child_cell.type = self.FC
        else:
            self.child_cell.type = self.LC
    

    
class NeighborTrackerPrinterSteppable(SteppableBasePy):

    # =============================
    # Lifecycle
    # =============================
    def __init__(self, frequency=10):
        super().__init__(frequency)
        # runtime buffers
        self.metrics_data = []
        self.boundary_data = []
        self.DefectorPosition_data = []
        self.TumorPosition_data = []

        # Persistent cluster tracking (stable IDs across time)
        self.prev_tracks = []           
        self.next_cluster_id = 1         
        self.centroid_match_radius = 12.0  

        self.prev_clusters_info = []  

    def start(self):
        self._init_outputs()
        self._init_plots()
        self.create_scalar_field_cell_level_py("myField")

    def step(self, mcs):
        final_step = self.simulator.getNumSteps()
        xmax, ymax = self.dim.x, self.dim.y

        tips = self.field.myField
        tips.clear()
        if mcs == 0:
            max_height = 40  
        min_height = float('inf')
        Tumor = nx.Graph()
        surface = []             # cells on surface (no neighbor on some face)
        clustercells = []        # members of non-tumor clusters

        # =============================
        # 1) MAIN TUMOR (BFS from substrate line y=1)
        # =============================
        Tumorcells = self._collect_main_tumor(Tumor, surface)
        min_tumor_y = self._lowest_y(Tumorcells)
        stalklc, stalkcells = self._count_stalk_leaders(Tumorcells)

        # =============================
        # 2) DEFECTORS / DETACHED (diagnostics)
        # =============================
        defectorcells = self._find_single_leader_defectors(min_tumor_y)
        min_height = self._scan_min_empty_height(min_height)
        detached_cells = self._find_detached_non_tumor_cells(Tumorcells, min_tumor_y, tips)
        num_defectors = len(detached_cells)
        if show_plots:
            self.plot_win2.add_data_point('defectors', mcs, len(defectorcells))
            self.plot_win2.add_data_point('defectors+clusters', mcs, num_defectors)

        # =============================
        # 3) CLUSTERS (outside tumor) + ID assignment
        # =============================
        cluster_compositions = self._collect_clusters_outside_tumor(Tumor, Tumorcells, clustercells)
        clusters = len(cluster_compositions)
        if cluster_compositions:
            # per-frame left→right rank (1..N)
            cluster_compositions.sort(key=lambda c: (c["centroid_x"], c["centroid_y"]))
            for i, c in enumerate(cluster_compositions, start=1):
                c["ClusterID_frame"] = i
            # stable IDs across time
            self._assign_stable_ids(cluster_compositions)

        curr_member_union = set().union(
            *(set(c["member_ids"]) for c in cluster_compositions)
        ) if cluster_compositions else set()

        # singletons = detached cells that are not members of any current cluster
        singleton_ids = set(detached_cells) - curr_member_union

        # detect merges / splits / dissolves
        merges, splits, dissolves, curr_info = self._detect_merge_split(
            self.prev_clusters_info, 
            cluster_compositions,
            curr_tumor_ids=set(Tumorcells),     # re-absorption into main tumor
            singleton_ids=singleton_ids         # dispersion to single cells
        )

        for evt in merges:
            self.events_writer.writerow([mcs, "merge", evt["parents"], [evt["child"]], evt["fractions"], "", ""])
        for evt in splits:
            self.events_writer.writerow([mcs, "split", [evt["parent"]], evt["children"], evt["fractions"], "", ""])
        for evt in dissolves:
            self.events_writer.writerow([
                mcs, "dissolve", [evt["parent"]], [], [], 
                round(evt["lost_to_singletons"],3), 
                round(evt["lost_to_main_tumor"],3)
            ])

        self.prev_clusters_info = curr_info
    
            
            

        # save cluster CSV every 10 MCS
        if mcs % 10 == 0:
            self.save_cluster_compositions(cluster_compositions, mcs)

        # =============================
        # 4) BOUNDARIES (main vs outermost)
        # =============================
        x_coords, main_tumor_y, outermost_y = self._scan_boundaries(xmax, ymax, Tumorcells, clustercells)
        x_array, main_tumor_array, outermost_array, lowest_point_array = \
            self._arrays_for_boundaries(x_coords, main_tumor_y, outermost_y)

        # =============================
        # 5) AREAS / FINGERS
        # =============================
        invasive_area, infiltrative_area = self._areas(x_array, main_tumor_array, outermost_array, lowest_point_array)
        finger_peaks, branches = self._fingers(main_tumor_array)
        if show_plots:
            self.plot_win1.add_data_point('Invasive Area', mcs, invasive_area)
            self.plot_win1.add_data_point('Infiltrative Area', mcs, infiltrative_area)

        # write metrics line
        self.metrics_writer.writerow([mcs, invasive_area, infiltrative_area, branches,
                                      len(defectorcells), num_defectors, clusters])

        # =============================
        # 6) STALKS & INVARIA (perimeter/complexity inputs)
        # =============================
        Surface = nx.subgraph(Tumor, surface)
        endpoints, epnodes = self._surface_endpoints(Surface, tips)
        stalks = len(endpoints)
        epheight, avgheight, heightvar, max_height = self._endpoint_heights(epnodes, min_height, stalks)
        min_height = self._update_min_height_with_surface(Surface, min_height)
        invarea = self._invasive_volume_minus_base(Tumor, clustercells, defectorcells, min_height)

        # =============================
        # 7) PERIODIC DETAILED OUTPUTS
        # =============================
        if mcs % 100 == 0 and len(x_array):
            boundary_data = [[x_array[j], main_tumor_array[j], outermost_array[j], lowest_point_array[j]]
                             for j in range(len(x_array))]
            defector_positions = self._positions(detached_cells)
            tumor_positions, tumor_leaders, tumor_followers = self._tumor_positions(Tumorcells)
            self.save_mcs_data(self.output_dir1, mcs,
                               boundary_data,
                               defector_positions,
                               tumor_positions,
                               tumor_leaders,
                               tumor_followers)

        # =============================
        # 8) FINAL SNAPSHOT (plots + text log)
        # =============================
        if mcs == final_step - 1:
            perimeter = self._perimeter(Surface)
            complexity = (np.square(perimeter) / (4 * np.pi * invarea)) if invarea > 0 else 0.0
            self._final_plots(x_coords, main_tumor_y, outermost_y,
                              x_array, main_tumor_array, lowest_point_array,
                              detached_cells, Tumorcells, finger_peaks)
            self._final_text_log(mcs, perimeter, complexity, endpoints,
                                 Tumor, invasive_area, infiltrative_area,
                                 len(defectorcells), num_defectors,
                                 branches, stalklc, avgheight, heightvar,
                                 max_height, min_height, clusters,
                                 cluster_compositions)

    def finish(self):
        if self.output_dir is not None and show_plots:
            png_output_path6 = Path(self.output_dir).joinpath("MainTumor_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+".png")
            self.plot_win6.save_plot_as_png(png_output_path6, 1000, 1000)
        try:
            self.events_file.close()
        except Exception:
            pass

    # =============================
    # Outputs / Plots
    # =============================
    def _init_outputs(self):
        now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        self.f = open(self.output_dir + f"/data_{Jlf}_{mu}_{PP}.txt", "a")
        self.metrics_file = open(os.path.join(self.output_dir, f"Metrics_Data_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(["MCS", "Invasive Area", "Infiltrative Area", "Fingers",
                                      "Single Defects", "Detached Cells", "Clusters"])
        self.output_dir1 = self.output_dir + f"/PositionData_{Jlf}_{mu}_{PP}"
        os.makedirs(self.output_dir1, exist_ok=True)

        # NEW: events CSV (merges/splits)
        self.events_file = open(os.path.join(self.output_dir, f"ClusterEvents_{Jlf}_{mu}_{PP}.csv"), "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(
            ["MCS","Event","Parents","Children","OverlapFractions","LostToSingletons","LostToMainTumor"]
        )


    def _init_plots(self):
        if not show_plots:
            return
        self.plot_win1 = self.add_new_plot_window(
            title='Areas Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS', y_axis_title='Area (micron^2)',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True}
        )
        self.plot_win2 = self.add_new_plot_window(
            title='Defectors Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS', y_axis_title='Number of Defectors',
            x_scale_type='linear', y_scale_type='linear', grid=False,
            config_options={'legend': True}
        )
        self.plot_win6 = self.add_new_plot_window(
            title='Main Tumor Boundary',
            x_axis_title='X Position', y_axis_title='Y Position',
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
        """
        Save BOTH IDs:
        - 'Cluster ID' (stable across time)
        - 'ClusterID_frame' (1..N within this frame by X)
        """
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
                      tumor_follower_cells):
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
            save_csv("BoundaryData", ["X", "Main_Tumor", "Outermost", "Lowest_boundary_point"], boundary_data)
        if defector_positions:
            save_csv("DefectorPosition", ["id", "x", "y"], defector_positions)
        if tumor_positions:
            save_csv("TumorPosition", ["id", "x", "y"], tumor_positions)
        if tumor_leader_cells:
            save_csv("TumorLeaderCells", ["id", "x", "y"], tumor_leader_cells)
        if tumor_follower_cells:
            save_csv("TumorFollowerCells", ["id", "x", "y"], tumor_follower_cells)

    # =============================
    # BFS / Core graph ops
    # =============================
    def _bfs(self, Tumor, surface, start_cell):
        """
        Breadth-first search from a starting cell.
        - Adds nodes/edges to Tumor graph with xCOM/yCOM & weights.
        - Accumulates surface IDs into 'surface' (no neighbor on some face).
        Returns: set of visited cell IDs.
        """
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
                    w = np.sqrt(np.square(m.xCOM - x) + np.square(m.yCOM - y))
                    Tumor.add_edge(m.id, neighbor.id, weight=w)
                if not neighbor:
                    if m.id not in surface:
                        surface.append(m.id)
        return visited

    # =============================
    # Main tumor / defectors
    # =============================
    def _collect_main_tumor(self, Tumor, surface):
        """Grow the main tumor set by BFS from the substrate line y=1."""
        Tumorcells = set()
        for x in range(0, self.dim.x - 1):
            cell0 = self.cell_field[x, 1, 0]
            if cell0 and cell0.id not in Tumorcells:
                Tumorcells.update(self._bfs(Tumor, surface, cell0))
        return Tumorcells

    def _lowest_y(self, Tumorcells):
        """Find lowest y among main tumor cells."""
        min_y = float('inf')
        for cid in Tumorcells:
            cell = self.fetch_cell_by_id(cid)
            if cell:
                min_y = min(min_y, cell.yCOM)
        return min_y

    def _count_stalk_leaders(self, Tumorcells):
        """Count LC inside the main tumor (stalk LC)."""
        stalklc = 0
        stalkcells = []
        for cell in self.cell_list_by_type(self.LC):
            if cell.id in Tumorcells:
                stalklc += 1
                stalkcells.append(cell)
        return stalklc, stalkcells

    def _find_single_leader_defectors(self, min_tumor_y):
        """Leaders with no neighbors above tumor baseline → single 'defectors' list."""
        defectorcells = []
        for cell in self.cell_list_by_type(self.LC):
            n = 0
            for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                if neighbor:
                    n += 1
            if n == 0 and cell.yCOM > min_tumor_y:
                defectorcells.append(cell.id)
        return defectorcells

    def _scan_min_empty_height(self, min_height):
        """Scan entire lattice for minimum empty y (kept as-is)."""
        for x, y, z in self.every_pixel():
            cell = self.cell_field[x, y, z]
            if not cell and y < min_height:
                min_height = y
        return min_height

    def _find_detached_non_tumor_cells(self, Tumorcells, min_tumor_y, tips):
        """
        Cells (LC/FC) not in main tumor, not adjacent to main tumor, and above the tumor baseline.
        """
        detached = []
        for cell in self.cell_list_by_type(self.LC, self.FC):
            if cell.id in Tumorcells:
                continue
            neighbors = [neigh for neigh, _ in self.get_cell_neighbor_data_list(cell) if neigh]
            if all(neigh.id not in Tumorcells for neigh in neighbors):
                if cell.yCOM > min_tumor_y:
                    detached.append(cell.id)
                    tips[cell] = 100
        return detached

    # =============================
    # Clusters outside tumor
    # =============================
    def _collect_clusters_outside_tumor(self, Tumor, Tumorcells, clustercells):
        """
        Build cluster_compositions (outside main tumor) by seeding BFS from FC only.
        (Logic preserved exactly.)
        """
        cluster_checked = set()
        cluster_compositions = []

        for cell in self.cell_list_by_type(self.FC):
            if cell.id in Tumorcells or cell.id in cluster_checked:
                continue
            new_cluster_cells = self._bfs(Tumor, [], cell)  # surface from cluster BFS is irrelevant downstream
            if len(new_cluster_cells) >= 2:  # keep >=2 to exclude singles
                cluster_checked.update(new_cluster_cells)
                clustercells.extend(new_cluster_cells)

                # counts
                num_leader_cells = sum(1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.LC)
                num_follower_cells = sum(1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.FC)

                # centroid
                xs = [self.fetch_cell_by_id(cid).xCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                ys = [self.fetch_cell_by_id(cid).yCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                centroid_x = float(np.mean(xs)) if xs else 0.0
                centroid_y = float(np.mean(ys)) if ys else 0.0

                # member coords (for CSV)
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
                    "member_ids": list(new_cluster_cells)  # NEW: for merge/split detection
                })

                if show_plots:
                    self.plot_win6.add_data_point("Cluster Centroids", centroid_x, centroid_y)

        return cluster_compositions

    # =============================
    # Boundary scanning & arrays
    # =============================
    def _scan_boundaries(self, xmax, ymax, Tumorcells, clustercells):
        """
        For each x, scan downward (y = ymax-1 .. 0) to find:
        - top_main_y (highest y that is part of Tumorcells and not in clustercells)
        - top_outer_y (highest y that is non-medium)
        """
        x_coords, main_tumor_y, outermost_y = [], [], []
        for x in range(xmax):
            top_main_y = None
            top_outer_y = None
            for y in range(ymax - 1, -1, -1):
                cell = self.cell_field[x, y, 0]
                if cell:
                    if top_outer_y is None and cell.type != self.MEDIUM:
                        top_outer_y = y
                    if cell.id in Tumorcells and top_main_y is None and cell.id not in clustercells:
                        top_main_y = y
                if top_outer_y is not None and top_main_y is not None:
                    break
            if top_main_y is not None and top_outer_y is not None:
                x_coords.append(x)
                main_tumor_y.append(top_main_y)
                outermost_y.append(top_outer_y)
        return x_coords, main_tumor_y, outermost_y

    def _arrays_for_boundaries(self, x_coords, main_tumor_y, outermost_y):
        x_array = np.array(x_coords)
        main_tumor_array = np.array(main_tumor_y)
        outermost_array = np.array(outermost_y)
        lowest_point_array = np.full_like(x_array, np.min(main_tumor_array)) if len(main_tumor_array) else np.array([])
        return x_array, main_tumor_array, outermost_array, lowest_point_array

    # =============================
    # Areas / Fingers
    # =============================
    def _areas(self, x, main_tumor, outermost, base):
        if len(x) == 0:
            return 0.0, 0.0
        invasive_area = trapz(main_tumor - base, x)
        infiltrative_area = trapz(outermost - base, x)
        return invasive_area, infiltrative_area

    def _fingers(self, main_tumor_array):
        if len(main_tumor_array) == 0:
            return np.array([]), 0
        raw_peaks, _ = find_peaks(main_tumor_array, prominence=10, distance=10, width=5)
        merged_peaks = []
        min_sep = 15
        last = -np.inf
        for p in raw_peaks:
            if p - last > min_sep:
                merged_peaks.append(p)
                last = p
        finger_peaks = np.array(merged_peaks)
        return finger_peaks, len(finger_peaks)

    # =============================
    # Stalks / perimeter / invarea
    # =============================
    def _surface_endpoints(self, Surface, tips):
        """
        Find graph endpoints on Surface (degree <= 2).
        Mark tips in the scalar field (same as original).
        """
        endpoints, epnodes = [], []
        for node in list(Surface.nodes):
            if Surface.degree(node) <= 2:
                cell = self.fetch_cell_by_id(node)
                if cell:
                    endpoints.append(cell)
                    epnodes.append(node)
                    tips[cell] = 100
        return endpoints, epnodes

    def _endpoint_heights(self, epnodes, min_height, stalks):
        epheight = [self.fetch_cell_by_id(node).yCOM - min_height
                    for node in epnodes if self.fetch_cell_by_id(node)]
        avgheight = np.mean(epheight) if stalks > 0 else 0
        heightvar = np.var(epheight) if stalks > 0 else 0
        max_height = np.amax(epheight) if stalks > 0 else 0
        return epheight, avgheight, heightvar, max_height

    def _update_min_height_with_surface(self, Surface, min_height):
        for id_ in Surface.nodes:
            cell = self.fetch_cell_by_id(id_)
            if cell and cell.yCOM < min_height:
                min_height = cell.yCOM
        return min_height

    def _invasive_volume_minus_base(self, Tumor, clustercells, defectorcells, min_height):
        invarea = 0
        for id_ in Tumor.nodes:
            cell = self.fetch_cell_by_id(id_)
            if cell and (id_ not in clustercells) and (id_ not in defectorcells):
                invarea += cell.volume
        invarea = invarea - (min_height * 500)
        return invarea

    def _perimeter(self, Surface):
        perimeter = 0
        for u, v, e in Surface.edges(data=True):
            perimeter += e['weight']
        return perimeter

    # =============================
    # Periodic data helpers
    # =============================
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

    # =============================
    # Final outputs
    # =============================
    def _final_plots(self, x_coords, main_tumor_y, outermost_y,
                     x_array, main_tumor_array, lowest_point_array,
                     detached_cells, Tumorcells, finger_peaks):
        if not show_plots or not len(x_coords):
            return
        for x, y in zip(x_coords, main_tumor_y):
            self.plot_win6.add_data_point("Tumor_Boundary", x, y)
        for x, y in zip(x_coords, outermost_y):
            self.plot_win6.add_data_point("Outer_Boundary", x, y)
            self.plot_win6.add_data_point("Outer_Boundary_Curve", x, y)
        for p in finger_peaks:
            self.plot_win6.add_data_point("Branch Points", x_array[p], main_tumor_array[p])
        for j in range(len(x_array)):
            self.plot_win6.add_data_point("Lowest Tumor Boundary Line", x_array[j], lowest_point_array[j])
        for cell_id in detached_cells:
            cell = self.fetch_cell_by_id(cell_id)
            if cell:
                self.plot_win6.add_data_point("Defected Cells", cell.xCOM, cell.yCOM)
        for cell_id in set(Tumorcells):
            cell = self.fetch_cell_by_id(cell_id)
            if cell:
                self.plot_win6.add_data_point("Main Tumor Cells", cell.xCOM, cell.yCOM)
                if cell.type == self.LC:
                    self.plot_win6.add_data_point("Tumor Leader Cells", cell.xCOM, cell.yCOM)
                elif cell.type == self.FC:
                    self.plot_win6.add_data_point("Tumor Follower Cells", cell.xCOM, cell.yCOM)

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
        self.f.write("Average Height of Stalks: " + str(avgheight) + "\n")
        self.f.write("Variance in Height of Stalks: " + str(heightvar) + "\n")
        self.f.write("max_height: " + str(max_height) + "\nmin_height: " + str(min_height) + "\n")
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

    # =============================
    # Stable ID tracker & merge/split
    # =============================
    def _euclid(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _assign_stable_ids(self, comps):
        """
        Assign persistent 'ClusterID_stable' to items in 'comps' based on nearest-centroid
        match to previous frame (thresholded). New clusters get N+1, N+2, ...
        """
        if not comps:
            return

        # First frame with clusters: sort by X (then Y) and assign 1..N
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
        """
        prev_info: [{'id': int, 'members': set(ids)}] from previous frame
        curr_comps: current frame cluster dicts (incl. 'ClusterID_stable' and 'member_ids')
        curr_tumor_ids: set of current main-tumor cell IDs
        singleton_ids:  set of detached cells that are not in any current cluster
        Returns:
          merges:   [{child, parents:[...], fractions:[...]}]
          splits:   [{parent, children:[...], fractions:[...]}]
          dissolves:[{parent, lost_to_singletons, lost_to_main_tumor}]
          curr_info:[{'id', 'members'}] for next frame
        """
        # current clusters as sets
        curr_info = [
            {"id": c.get("ClusterID_stable"), "members": set(c.get("member_ids", []))}
            for c in curr_comps if c.get("ClusterID_stable") is not None
        ]

        def jacc(a, b):
            if not a or not b: return 0.0
            inter = len(a & b); uni = len(a | b)
            return 0.0 if uni == 0 else inter/uni

        merges, splits, dissolves = [], [], []

        # MERGES: many prev -> one curr
        for c in curr_info:
            parents, fracs = [], []
            for p in prev_info:
                J = jacc(p["members"], c["members"])
                cover_child = len(p["members"] & c["members"]) / max(1, len(c["members"]))
                if J >= jaccard_min and cover_child >= cover_min:
                    parents.append(p["id"]); fracs.append(round(cover_child,3))
            if len(parents) >= 2:
                merges.append({"child": c["id"], "parents": parents, "fractions": fracs})

        # SPLITS: one prev -> many curr
        for p in prev_info:
            children, fracs = [], []
            for c in curr_info:
                J = jacc(p["members"], c["members"])
                cover_parent = len(p["members"] & c["members"]) / max(1, len(p["members"]))
                if J >= jaccard_min and cover_parent >= cover_min:
                    children.append(c["id"]); fracs.append(round(cover_parent,3))
            if len(children) >= 2:
                splits.append({"parent": p["id"], "children": children, "fractions": fracs})
            elif len(children) == 0:
                # DISSOLVE / DISAPPEAR: no child clusters this frame
                parent_sz = max(1, len(p["members"]))
                lost_to_singletons = len(p["members"] & singleton_ids) / parent_sz
                lost_to_main = len(p["members"] & curr_tumor_ids) / parent_sz
                dissolves.append({
                    "parent": p["id"],
                    "lost_to_singletons": lost_to_singletons,
                    "lost_to_main_tumor": lost_to_main
                })

        return merges, splits, dissolves, curr_info
