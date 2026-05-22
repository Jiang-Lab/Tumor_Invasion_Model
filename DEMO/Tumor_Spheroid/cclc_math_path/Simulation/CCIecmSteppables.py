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


k = 25 #(25) <1-10,10-80> This is the percent concentration of Leaders in the tumor
matrix = np.zeros((300,500), int)
lccelldiv=0
fccelldiv=0
divtime=0
fgrow=0.015 #(.11) <.005 to .015> growth rate for Followers 
lgrow = .010 #(.015) growth rate for Leaders (less than Followers)


Jlf = 2 #The Adhesion (contact energy) of leader cells and follower cells
mu = 24  #Chemotaxis Lambda
PP = 0.5 # the percentage of the follower cells allowed to proliferate

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
            x1 = np.random.randint(200,300)
            y1 = np.random.randint(200,300)
            c1 = self.cellField[x1,y1,0]
            if c1 and c1.type == self.FC:
                self.cellField[x1,y1,0] = lc #change
                i= len(self.cell_list_by_type(self.LC))/(len(self.cell_list_by_type(self.LC))+len(self.cell_list_by_type(self.FC)))

        
        mv = self.field.MV
        center_x, center_y = 250, 250  

        max_radius = np.sqrt(center_x**2 + center_y**2)

        for x in range(0, 500, 1):
            for y in range(0, 500):  
                g = 1  

                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                mv[x, y, 0] = (distance / g)

 
        for cell in self.cell_list_by_type(self.FC, self.LC):
            cell.targetVolume = 10
            cell.lambdaVolume = 2.0

        self.get_xml_element("J_LF").cdata = Jlf
        self.get_xml_element("lambda_chem").Lambda = mu
        

        
    def step(self, mcs): 
        
        Leaders = str(len(self.cell_list_by_type(self.LC)))
        Followers = str(len(self.cell_list_by_type(self.FC)))
        Total = str(len(self.cell_list))
        
        self.cellcount_writer.writerow([mcs, Leaders, Followers, Total])
 
        if mcs == 700:
            self.cellcount_file.close()
        
class GrowthSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self, frequency)
       
    def step(self, mcs):
        global fgrow
        
        for cell in self.cell_list_by_type(self.FC):
            if cell.targetVolume<20:
                cell.targetVolume += fgrow 
        
class NeighborTrackerPrinterSteppable(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)
        self.metrics_data = []
        self.boundary_data = []
        self.DefectorPosition_data = []
        self.TumorPosition_data = []
        
    def start(self):
                

        now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        self.f = open(self.output_dir +"/data_" +  str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".txt", "a")
        
        self.metrics_file = open(os.path.join(self.output_dir, f"Metrics_Data_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.metrics_writer = csv.writer(self.metrics_file)
        self.metrics_writer.writerow(["MCS", "Invasive Area", "Infiltrative Area", "Fingers", "Single Defects", "Detached Cells", "Clusters"])
        
        
        self.output_dir1 = self.output_dir + f"/PositionData_{Jlf}_{mu}_{PP}"
        os.makedirs(self.output_dir1, exist_ok=True)
        
        self.boundary_file = open(os.path.join(self.output_dir1, f"BoundaryData_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.boundary_writer = csv.writer(self.boundary_file)
        self.boundary_writer.writerow(["X", "Main_Tumor", "Outermost", "Lowesr_boundary_point"])

        self.DefectorPosition_file = open(os.path.join(self.output_dir1, f"DefectorPosition_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.DefectorPosition_writer = csv.writer(self.DefectorPosition_file)
        self.DefectorPosition_writer.writerow(["id", "x", "y"])
        
        self.TumorPosition_file = open(os.path.join(self.output_dir1, f"TumorPosition_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.TumorPosition_writer = csv.writer(self.TumorPosition_file)
        self.TumorPosition_writer.writerow(["id", "x", "y"])
        
        self.TumorLeader_file = open(os.path.join(self.output_dir1, f"TumorLeaderCells_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.TumorLeader_writer = csv.writer(self.TumorLeader_file)
        self.TumorLeader_writer.writerow(["id", "x", "y"])

        self.TumorFollower_file = open(os.path.join(self.output_dir1, f"TumorFollowerCells_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.TumorFollower_writer = csv.writer(self.TumorFollower_file)
        self.TumorFollower_writer.writerow(["id", "x", "y"])

        self.ClusterComposition_file = open(os.path.join(self.output_dir1, f"ClusterComposition_{Jlf}_{mu}_{PP}.csv"), "a", newline="")
        self.ClusterComposition_writer = csv.writer(self.ClusterComposition_file)
        self.ClusterComposition_writer.writerow(["Cluster ID", "Leader Cells", "Follower Cells", "Total Cells", "Centroid_X", "Centroid_Y"])

        
        
        self.plot_win1 = self.add_new_plot_window(
            title='Areas Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Area (micron^2)',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False, 
            config_options={'legend':True} 
        )
        self.plot_win2 = self.add_new_plot_window(
            title='Defectors Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Number of Defectors',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False, 
            config_options={'legend':True} 
        )
        
        
        self.plot_win6 = self.add_new_plot_window(
            title='Main Tumor Boundary',
            x_axis_title='X Position',
            y_axis_title='Y Position',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False,
            config_options={'legend':True} 
        )
        
        
        self.plot_win1.add_plot("Invasive Area", style='Lines', color='yellow', size=2)
        self.plot_win1.add_plot("Infiltrative Area", style='Lines', color='red', size=2)
        self.plot_win2.add_plot("defectors", style='Lines', color='green', size=5)
        self.plot_win2.add_plot("defectors+clusters", style='Lines', color='blue', size=2)
        #self.plot_win3.add_histogram_plot(plot_name='Cluster Composition', color='green', alpha=100)
        
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

        
        self.create_scalar_field_cell_level_py("myField")  

    def step(self, mcs):
        
        
        xmax, ymax = 500, 300
#====================================================================================================================   
#       Breadth First Search (BFS)
#====================================================================================================================          
        tips = self.field.myField
        tips.clear()

        if mcs == 0:
            max_height = 40
        min_height = float('inf')
        
        queue, tumorcells, stalkcells, surface, endpoints = [], [], [], [], []
        Tumor = nx.Graph()
        Tumor.clear()

        def bfs(visited, node):  
            visited.add(node.id)
            queue.append(node.id)
            Tumor.add_node(node.id, xCOM=node.xCOM, yCOM=node.yCOM) 

            while queue:  # Creating loop to visit each node
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
                        if m not in surface:
                            surface.append(m.id)
            return visited

#================================================================================================================================ 
#               Core Tumor Body
#================================================================================================================================ 

        Tumorcells = set()  # List for visited nodes.
        clustercells = []
        Queue = []  
        clusters = 0
        stalklc = 0
        min_tumor_y = float('inf')

        for x in range(0, 499):
            cell0 = self.cell_field[x, 1, 0]
            if cell0:
                if cell0.id not in Tumorcells:
                    Tumorcells.update(bfs(set(), cell0))
                    #Tumorcells += bfs(Tumorcells, cell0)
                    
        # Find the lowest point of the tumor cluster
        for cell_id in Tumorcells:
            cell = self.fetch_cell_by_id(cell_id)
            if cell:
                min_tumor_y = min(min_tumor_y, cell.yCOM)
            

        for cell in self.cell_list_by_type(self.LC):
            if cell.id in Tumorcells:
                stalklc += 1
                stalkcells.append(cell)
        
       
        defectorcells = []
        for cell in self.cell_list_by_type(self.LC):
            n = 0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if neighbor:
                    n += 1
            if n == 0:
                if cell.yCOM > min_tumor_y:
                    defectorcells.append(cell.id)

        for x, y, z in self.every_pixel():
            cell = self.cell_field[x, y, z]
            if not cell and y < min_height:
                min_height = y

        
        # Identify defectors: cells that are NOT in the main tumor
        detached_cells = []
        for cell in self.cell_list_by_type(self.LC, self.FC):
            if cell.id not in Tumorcells:  # Cells that are NOT part of the main tumor
                neighbors = [neighbor for neighbor, _ in self.get_cell_neighbor_data_list(cell) if neighbor]
                
                # Check if surrounded only by medium or other defectors
                if all(neigh.id not in Tumorcells for neigh in neighbors):
                    if cell.yCOM > min_tumor_y :
                        detached_cells.append(cell.id)
                        tips[cell] = 100  
        
        num_defectors = len(detached_cells) 

        
        self.plot_win2.add_data_point('defectors', mcs, len(defectorcells))
        self.plot_win2.add_data_point('defectors+clusters', mcs, num_defectors)
#======================================================================================================================================  
#                       Clusters
#======================================================================================================================================  
        cluster_checked = set()
        cluster_compositions = []
        
        for cell in self.cell_list_by_type(self.FC):
            if cell.id not in Tumorcells and cell.id not in cluster_checked:
                new_cluster_cells = bfs(set(), cell)
                num_leader_cells = sum(1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.LC)
                if len(new_cluster_cells) >= 2 and num_leader_cells >= 1:
                    cluster_checked.update(new_cluster_cells)
                    clustercells.extend(new_cluster_cells)
                    clusters += 1

                    num_follower_cells = sum(1 for cid in new_cluster_cells if self.fetch_cell_by_id(cid).type == self.FC)
                    x_coordinates = [self.fetch_cell_by_id(cid).xCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                    y_coordinates = [self.fetch_cell_by_id(cid).yCOM for cid in new_cluster_cells if self.fetch_cell_by_id(cid)]
                    centroid_x = np.mean(x_coordinates) if x_coordinates else 0
                    centroid_y = np.mean(y_coordinates) if y_coordinates else 0

                    if mcs == 700:
                        self.ClusterComposition_writer.writerow([
                            clusters, num_leader_cells, num_follower_cells, len(new_cluster_cells), centroid_x, centroid_y
                        ])
                        self.plot_win6.add_data_point("Cluster Centroids", centroid_x, centroid_y)

                    cluster_compositions.append({
                        "cluster_id": clusters,
                        "leader_cells": num_leader_cells,
                        "follower_cells": num_follower_cells,
                        "total_cells": len(new_cluster_cells)
                    })

#=================================================================================================================================  
#                      Core Tumor Boundary and Outermost Boundary
#=================================================================================================================================  
 
        x_coords , main_tumor_y, outermost_y  = [], [], []

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

#============================================================================================================================== 
#          Invasive Area, Infiltrative Area, & Number of Branch Points (fingers)
#==============================================================================================================================   
        x_array = np.array(x_coords)
        main_tumor_array = np.array(main_tumor_y)
        outermost_array = np.array(outermost_y)
        lowest_point_array = np.full_like(x_array, np.min(main_tumor_array))    
            
        invasive_area = trapz(main_tumor_array - lowest_point_array, x_array)
        infiltrative_area = trapz(outermost_array - lowest_point_array, x_array)    
            
        raw_peaks, _ = find_peaks(main_tumor_array, prominence=10, distance=10, width=5)
        merged_peaks = []
        min_sep = 15
        last = -np.inf
        for p in raw_peaks:
            if p - last > min_sep:
                merged_peaks.append(p)
                last = p
        finger_peaks = np.array(merged_peaks)
        branches = len(finger_peaks)
        
        
        self.plot_win1.add_data_point('Invasive Area', mcs, invasive_area)
        self.plot_win1.add_data_point('Infiltrative Area', mcs, infiltrative_area)
 
        self.metrics_writer.writerow([mcs, invasive_area, infiltrative_area, branches, len(defectorcells), num_defectors, clusters])
              
#======================================================================================================================================   
#               Stalks and Area
#======================================================================================================================================  
        invarea = 0
        Surface = nx.subgraph(Tumor, surface)
        xmin = 10
        for id in surface:
            cell = self.fetch_cell_by_id(id)
            if cell.xCOM < xmin:
                xmin = cell.xCOM
                startcell = cell.id
        
        vsurface, queue, endpoints, epnodes, epheight = [], [],  [], [], []
        vsurface.append(startcell)
        queue.append(startcell)
                
        for node in list(Surface.nodes):
            if Surface.degree(node) <= 2:  
                cell = self.fetch_cell_by_id(node)
                if cell:
                    endpoints.append(cell)
                    epnodes.append(node)
                    tips[cell] = 100  
        stalks = len(endpoints)
     
        epheight = [self.fetch_cell_by_id(node).yCOM - min_height for node in epnodes if self.fetch_cell_by_id(node)]
        avgheight = np.mean(epheight) if stalks > 0 else 0
        heightvar = np.var(epheight) if stalks > 0 else 0
        max_height = np.amax(epheight) if stalks > 0 else 0

        
        for id in Surface.nodes:
            cell = self.fetch_cell_by_id(id)
            if cell and cell.yCOM < min_height:
                min_height = cell.yCOM
        for id in Tumor.nodes:
            cell = self.fetch_cell_by_id(id)
            if cell:
                if cell.id not in clustercells:
                    if cell.id not in defectorcells:
                        invarea += cell.volume

        invarea = invarea - (min_height * 500)

#============================================================================================================================== 
#       At MCS = 700
#==============================================================================================================================   
        if mcs == 700:

            perimeter = 0
            for u,v,e in Surface.edges(data=True):
                perimeter += e['weight']
            complexity = np.square(perimeter)/(4*np.pi*invarea)
            
            for x, y in zip(x_coords, main_tumor_y):
                self.plot_win6.add_data_point("Tumor_Boundary", x, y)

            for x, y in zip(x_coords, outermost_y):
                self.plot_win6.add_data_point("Outer_Boundary", x, y)
                self.plot_win6.add_data_point("Outer_Boundary_Curve", x, y)
                
            for p in finger_peaks:
                self.plot_win6.add_data_point("Branch Points", x_array[p], main_tumor_array[p])
                
            
            for j in range(len(x_array)):
                self.boundary_writer.writerow([x_array[j], main_tumor_array[j], outermost_array[j], lowest_point_array[j]])
                self.plot_win6.add_data_point("Lowest Tumor Boundary Line", x_array[j], lowest_point_array[j])
            
            for cell_id in detached_cells:
                cell = self.fetch_cell_by_id(cell_id)
                if cell:
                    self.plot_win6.add_data_point("Defected Cells", cell.xCOM, cell.yCOM)
                    self.DefectorPosition_writer.writerow([cell_id, cell.xCOM, cell.yCOM])
  
            for cell_id in set(Tumorcells):  
                cell = self.fetch_cell_by_id(cell_id)
                if cell:
                    self.plot_win6.add_data_point("Main Tumor Cells", cell.xCOM, cell.yCOM)     
                    self.TumorPosition_writer.writerow([cell_id, cell.xCOM, cell.yCOM])
                    
                    if cell.type == self.LC:
                        self.plot_win6.add_data_point("Tumor Leader Cells", cell.xCOM, cell.yCOM)
                        self.TumorLeader_writer.writerow([cell_id, cell.xCOM, cell.yCOM])
                    elif cell.type == self.FC:
                        self.plot_win6.add_data_point("Tumor Follower Cells", cell.xCOM, cell.yCOM)
                        self.TumorFollower_writer.writerow([cell_id, cell.xCOM, cell.yCOM])
            
                 
            self.f.write("Step: " + str(mcs) + "\n")
            self.f.write("Tumor perimeter: " + str(perimeter) + "\n")
            self.f.write("Tumor complexity: " + str(complexity) + "\n")         
            self.f.write("Tumor endpoints(stalks): " + str(len(endpoints)) + "\n")
            self.f.write("Tumor cells: " + str(len(list(Tumor.nodes))) + "\n") # The number of cells in the main tumor (Not defected)

            self.f.write("Invasive Area:" + str(invasive_area) + "\n")          
            self.f.write("Infiltrative Area:" + str(infiltrative_area) + "\n")
            self.f.write("Single Defects: " + str(len(defectorcells)) + "\n")
            self.f.write("Detached Cells: " + str(num_defectors) + "\n")
            self.f.write("Branches: " + str(branches) + "\n")
            
            self.f.write("Stalk LC: " + str(stalklc) +"\n")
            self.f.write("Average Height of Stalks: " + str(avgheight) + "\n")
            self.f.write("Variance in Height of Stalks: " + str(heightvar) + "\n")
            self.f.write("max_height: " + str(max_height) +"\nmin_height: " + str(min_height) + "\n")
            

            self.f.write("CLUSTER DATA:\n Total Clusters: " + str(clusters) +"\n")
            self.f.write("Cluster Composition:"+ "\n")
            for cluster in cluster_compositions:
                self.f.write(f"Cluster {cluster['cluster_id']} - Leader Cells: {cluster['leader_cells']}, Follower Cells: {cluster['follower_cells']}, Total Cells: {cluster['total_cells']}\n")  
          
            self.f.write("END")
            self.f.close()
            
            self.metrics_file.close()
            self.boundary_file.close()
            self.DefectorPosition_file.close()
            self.TumorPosition_file.close()
            self.TumorLeader_file.close()
            self.TumorFollower_file.close()
            self.ClusterComposition_file.close()

    def finish(self):     
        if self.output_dir is not None:
            png_output_path6 = Path(self.output_dir).joinpath("MainTumor_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".png")
            self.plot_win6.save_plot_as_png(png_output_path6, 1000, 1000)
            

        return
     

   
   
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
                cell.dict["clock"] = None  # Set clock to None if not selected for proliferation

    def step(self, mcs):
        cells_to_divide = []
        global lccelldiv, fccelldiv
        
        for cell in self.cell_list_by_type(self.FC):
            # Only consider cells that have a clock (ignore cells with clock == None)
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
        self.parent_cell.dict["clock"] = 0  # Reset the parent's clock after division

        # Clone the parent's attributes to the child
        self.clone_parent_2_child()

        # Set the type of the child cell (same as the parent)
        if self.parent_cell.type == self.FC:
            self.child_cell.type = self.FC
        else:
            self.child_cell.type = self.LC

    