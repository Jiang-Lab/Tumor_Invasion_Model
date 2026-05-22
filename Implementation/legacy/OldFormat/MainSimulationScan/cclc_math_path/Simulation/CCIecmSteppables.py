from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup

from cc3d.core.PySteppables import *
from datetime import datetime
import numpy as np
import networkx as nx
from scipy.signal import argrelextrema
from pathlib import Path
from itertools import product
from scipy.interpolate import make_interp_spline



k = 25 #(25) <1-10,10-80> This is the percent concentration of Leaders in the tumor
matrix = np.zeros((300,500), int)
lccelldiv=0
fccelldiv=0
divtime=0
fgrow=0.015 #(.11) <.005 to .015> growth rate for Followers 
lgrow = .010 #(.015) growth rate for Leaders (less than Followers)


Jlf = 2.0 #The Adhesion (contact energy) of leader cells and follower cells
mu = 25 #Chemotaxis Lambda
PP = 0.5 # the percentage of the follower cells allowed to proliferate


iteration = {{iteration}}
parameter_values = {
    'Jlf': [-5,-4,-3,-2,-1,0,1,2,3,4,5],
    'mu': [0, 3, 6, 9, 12, 15, 18, 21, 24, 27,30],
    'PP': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
}
parameter_combinations = list(product(parameter_values['Jlf'], parameter_values['mu'], parameter_values['PP']))


index = iteration % len(parameter_combinations) 
if 0 <= index < len(parameter_combinations):
    Jlf, mu, PP = parameter_combinations[index]


class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency=1): 
        SteppableBasePy.__init__(self,frequency)
    
    def start(self):
        
        #try: 
        global tracklc, k
        self.m = open(self.output_dir +"/CellCount_"+ str(Jlf)+"_"+str(mu)+"_"+str(PP)+".txt", "a")
        iteration = self.param_scan_iteration

        i= len(self.cell_list_by_type(self.LC))/(len(self.cell_list_by_type(self.LC))+len(self.cell_list_by_type(self.FC)))
        
        #This is for creating k% concentration of leader cells
        while i < k/100:
            lc = self.new_cell(self.LC)
            rand = np.random
            x1 = np.random.randint(1,499)
            y1 = np.random.randint(1,19)
            c1 = self.cellField[x1,y1,0]
            if c1.type == 2:
                self.cellField[x1,y1,0] = lc #change
                i= len(self.cell_list_by_type(self.LC))/(len(self.cell_list_by_type(self.LC))+len(self.cell_list_by_type(self.FC)))

   
        mv = self.field.MV
        for x in range(0, 500, 1):
            for i in range(0,500):
                #mv[x, i, :] = x+i
                g=1 # (1) <0.1 to 15.0> raise this number to weaken the gradient field, or lower g to strengthen it
                mv[x, i, :] = i/g  
            


        for cell in self.cell_list_by_type(self.FC):

            cell.targetVolume = 10
            cell.lambdaVolume = 2.0
            
        for cell in self.cell_list_by_type(self.LC):

            cell.targetVolume = 10
            cell.lambdaVolume = 2.0
        
        
        # Xml parameters defined in the steppable
        J_LF = self.get_xml_element("J_LF")
        J_LF.cdata = Jlf    
        lambda_chem = self.get_xml_element("lambda_chem")
        lambda_chem.Lambda = mu
        

        
    def step(self, mcs): 

        self.m.write("Step: " +str(mcs)+ "\n")
        self.m.write("LC:" + str(len(self.cell_list_by_type(self.LC))) + "\n")
        self.m.write("FC:" + str(len(self.cell_list_by_type(self.FC))) + "\n")
 
        if mcs == 700:

            self.m.write("Final number of Cells: " +"\n")
            self.m.write("Final LC: " + str(len(self.cell_list_by_type(self.LC))) + "\n")
            self.m.write("Final FC: " + str(len(self.cell_list_by_type(self.FC))) + "\n")

            self.m.close()
        
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
 
    def start(self):
                

        now = datetime.now().strftime("%d_%m_%Y %H_%M_%S")
        self.f = open(self.output_dir +"/data_" +  str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".txt", "a")
        
        self.plot_win1 = self.add_new_plot_window(
            title='Invasive Area Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Invasive AREA',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False 
        )
        self.plot_win2 = self.add_new_plot_window(
            title='Defectors Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Number of Defectors',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False 
        )
        
        self.plot_win4 = self.add_new_plot_window(
            title='Infiltrative Area Over Time',
            x_axis_title='MonteCarlo Step (MCS)',
            y_axis_title='Area', 
            x_scale_type='linear', 
            y_scale_type='linear',
            grid=False
        )

        
        self.plot_win5 = self.add_new_plot_window(
            title='Outer Boundary Shape',
            x_axis_title='x area covered',
            y_axis_title='y area covered', 
            x_scale_type='linear', 
            y_scale_type='linear',
            grid=False
        )
        
        
        self.plot_win6 = self.add_new_plot_window(
            title='Main Tumor Boundary',
            x_axis_title='X Position',
            y_axis_title='Y Position',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False
        )
        
        
        self.plot_win1.add_plot("Invasive AREA", style='Lines', color='red', size=2)
        self.plot_win2.add_plot("defectors", style='Dots', color='green', size=5)
        self.plot_win2.add_plot("defectors+clusters", style='Lines', color='red', size=2)
        #self.plot_win3.add_histogram_plot(plot_name='Cluster Composition', color='green', alpha=100)
        self.plot_win4.add_plot("AUC", style='Lines', color='red', size=2)
        
        self.plot_win5.add_plot("Outer_Boundary", style='Dots', color='red', size=2)
        self.plot_win5.add_plot("Outer_Boundary_Curve", style='Lines', color='orange', size=2)
        
        self.plot_win6.add_plot("Tumor_Boundary", style='Lines', color='yellow', size=2)
        self.plot_win6.add_plot("Lowest Tumor Boundary Line", style='Lines', color='green', size=2)
        self.plot_win6.add_plot("Branch Points", style='Dots', color='red', size=5) 
        self.plot_win6.add_plot("Defected Cells", style="Dots", color="blue", size=3)
             
        
        
        self.create_scalar_field_cell_level_py("myField")  

    def step(self, mcs):
        global max, min
        tips = self.field.myField
        tips.clear()

        if mcs == 0:
            max_height = 40

        queue = []
        tumorcells = []
        stalkcells = []
        min_height = float('inf')
        surface = []
        Tumor = nx.Graph()
        Tumor.clear()
        endpoints = []
        
#====================Breadth First Search (BFS)======================================       
        def bfs(visited, node):  
            visited.append(node.id)
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
                            visited.append(neighbor.id)
                            queue.append(neighbor.id)
                            Tumor.add_node(neighbor.id, xCOM=x, yCOM=y)
                        w = np.sqrt(np.square(m.xCOM - x) + np.square(m.yCOM - y))
                        Tumor.add_edge(m.id, neighbor.id, weight=w)
                    if not neighbor:
                        if m not in surface:
                            surface.append(m.id)
            return visited

#====================Main Tumor Body==================================================

        Tumorcells = []  # List for visited nodes.
        clustercells = []
        Queue = []  
        clusters = 0
        stalklc = 0
        min_tumor_y = float('inf')

        for x in range(0, 499):
            cell0 = self.cell_field[x, 1, 0]
            if cell0:
                if cell0.id not in Tumorcells:
                    Tumorcells += bfs(Tumorcells, cell0)
                    
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

#====================Clusters========================================================        
        cluster_compositions = []
        for cell in self.cell_list_by_type(self.FC):
            if cell.id not in Tumorcells:
                clusters += 1  
                new_cluster_cells = bfs([], cell)  
                Tumorcells.extend(new_cluster_cells)

                # Now count leader and follower cells in each cluster
                num_leader_cells = sum(1 for cell_id in new_cluster_cells if self.fetch_cell_by_id(cell_id).type == self.LC)
                num_follower_cells = sum(1 for cell_id in new_cluster_cells if self.fetch_cell_by_id(cell_id).type == self.FC)

                # Store the composition of each clusters
                cluster_compositions.append({
                    "cluster_id": clusters,
                    "leader_cells": num_leader_cells,
                    "follower_cells": num_follower_cells,
                    "total_cells": len(new_cluster_cells)
                })    
  
        clusterfc = 0
        clusterlc = []
        invarea = 0
        Surface = nx.subgraph(Tumor, surface)
        xmin = 10
        for id in surface:
            cell = self.fetch_cell_by_id(id)
            if cell.xCOM < xmin:
                xmin = cell.xCOM
                startcell = cell.id
        
        for cell in self.cell_list_by_type(self.LC):
            if cell.id in clustercells:
                clusterlc.append(cell)
        for cell in self.cell_list_by_type(self.FC):
            if cell.id in clustercells:
                clusterfc += 1
        
 
#====================Stalks and Area========================================================  
        vsurface = []
        queue = []
        vsurface.append(startcell)
        queue.append(startcell)
        endpoints = []
        epnodes = []
        epheight = []
        
                
        for node in list(Surface.nodes):
            if Surface.degree(node) <= 2:  
                cell = self.fetch_cell_by_id(node)
                if cell:
                    endpoints.append(cell)
                    epnodes.append(node)
                    tips[cell] = 100  
        stalks = len(endpoints)
     
        epheight = [self.fetch_cell_by_id(node).yCOM - min_height for node in epnodes if self.fetch_cell_by_id(node)]
 
        
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
        avgheight = 0
        heightvar = 0
        if stalks > 0:
            avgheight = np.mean(epheight)
            heightvar = np.var(epheight)
            max_height = np.amax(epheight)
            
        self.plot_win1.add_data_point('Invasive AREA', mcs, invarea)
            
#=========================Infiltrative Area===========================================    
        
        xmax, ymax = 500, 300
        outer_boundary = [] 

        # Iterate through columns to find the topmost non-medium cell
        for x in range(xmax):
            for y in range(ymax - 1, -1, -1):  
                cell = self.cell_field[x, y, 0]
                if cell and cell.type != self.MEDIUM:
                    outer_boundary.append((x, y))
                    break  
            else:
                continue  

        if outer_boundary:
            outer_boundary = sorted(outer_boundary, key=lambda p: p[0])
            x_coords, y_coords = zip(*outer_boundary)
            
            upper_boundary = {}     # Identify the upper boundary points
            for x, y in outer_boundary:
                if x not in upper_boundary or y > upper_boundary[x]:
                    upper_boundary[x] = y
            upper_x = sorted(upper_boundary.keys())
            upper_y = [upper_boundary[x] for x in upper_x]
            
            
            min_outer_boundary_y = min(y_coords)        # Determine the lowest tumor boundary height

            # Calculate the infiltrative area (AUC) using the trapezoidal rule               
            auc = np.trapz(np.array(y_coords) - min_outer_boundary_y, x_coords) 
        else:
            auc = 0
          
         
        self.plot_win4.add_data_point("AUC", mcs, auc)    
    
#=========================Invasive Area==============================================  
        tumor_cells = []  
        for x in range(0, 499):
            cell0 = self.cell_field[x, 1, 0]
            if cell0:
                if cell0.id not in tumor_cells:
                    new_cluster = bfs([], cell0)
                    if len(new_cluster) > len(tumor_cells):
                        tumor_cells = new_cluster

        # Identify boundary of the main tumor
        boundary_points = []
        xmax, ymax = 500, 300
        for x in range(xmax):
            for y in range(ymax - 1, -1, -1):
                cell = self.cell_field[x, y, 0]
                if cell and cell.id in tumor_cells:
                    boundary_points.append((x, y))
                    break

        if boundary_points:
            boundary_points = sorted(boundary_points, key=lambda p: p[0])
            x_coords, y_coords = zip(*boundary_points)
            
            lowest_tumor_boundary_y = min(y_coords)  
            # Interpolate the smooth tumor boundary
            x_smooth = np.linspace(min(x_coords), max(x_coords), 200)
            spline = make_interp_spline(x_coords, y_coords, k=3)
            y_smooth = spline(x_smooth)
            
            # Compute invasive area using AUC (subtracting minimum baseline)
            AUC = np.trapz(np.array(y_smooth) - lowest_tumor_boundary_y, x_smooth)  # Trapezoidal Rule
            
            # Compute second derivative (curvature) to detect turning points
            y_smooth = np.array(y_smooth)
            second_derivative = np.gradient(np.gradient(y_smooth))

            # Identify turning points where the second derivative changes sign
            turning_points = np.where(np.diff(np.sign(second_derivative)))[0]
            concave_turning_points = [i for i in turning_points if second_derivative[i] < 0]
            
            # Filter turning points that meet the height requirement
            min_finger_height = 10  # Minimum height above lowest tumor boundary
            min_finger_width = 12     # Minimum x-distance between adjacent peaks

            valid_branches = []
            for i in range(len(concave_turning_points)):
                idx = concave_turning_points[i]

                # Ensure the protrusion has a minimum height difference
                if (y_smooth[idx] - lowest_tumor_boundary_y) >= min_finger_height:
                    # Ensure the protrusion is not just a minor wiggle (check width)
                    if i == 0 or (x_smooth[idx] - x_smooth[concave_turning_points[i - 1]]) > min_finger_width:
                        valid_branches.append(idx)

            branches = len(valid_branches)

      
       
        if mcs == 700:

            perimeter = 0
            for u,v,e in Surface.edges(data=True):
                perimeter += e['weight']
            complexity = np.square(perimeter)/(4*np.pi*invarea)
            
              
            if outer_boundary:
                # Plot the raw outer boundary points
                #for x, y in zip(x_coords, y_coords):
                    #self.plot_win5.add_data_point("Outer_Boundary", x, y)
                #Plot the upper boundary curve
                for x, y in zip(upper_x, upper_y):
                    self.plot_win5.add_data_point("Outer_Boundary_Curve", x, y)
            
            if boundary_points:
                # Plot tumor boundary
                for x, y in zip(x_smooth, y_smooth):
                    self.plot_win6.add_data_point("Tumor_Boundary", x, y)

                # Plot horizontal reference line at lowest tumor boundary height
                for x in range(min(x_coords), max(x_coords)):
                    self.plot_win6.add_data_point("Lowest Tumor Boundary Line", x, lowest_tumor_boundary_y)

                # Plot valid branches as red dots
                for i in valid_branches:
                    self.plot_win6.add_data_point("Branch Points", x_smooth[i], y_smooth[i])
                    
            for cell_id in detached_cells:
                cell = self.fetch_cell_by_id(cell_id)
                if cell:
                    self.plot_win6.add_data_point("Defected Cells", cell.xCOM, cell.yCOM)
            
            self.f.write("Step: " + str(mcs) + "\n")
            self.f.write("Tumor perimeter: " + str(perimeter) + "\n")
            self.f.write("Tumor complexity: " + str(complexity) + "\n")         
            self.f.write("Tumor endpoints(stalks): " + str(len(endpoints)) + "\n")
            self.f.write("Tumor cells: " + str(len(list(Tumor.nodes))) + "\n") # The number of cells in the main tumor (Not defected)

            self.f.write("Invasive Area:" + str(AUC) + "\n")          
            self.f.write("Infiltrative Area:" + str(auc) + "\n")
            
            self.f.write("Single Defects: " + str(len(defectorcells)) + "\n")
            self.f.write("Branches: " + str(branches) + "\n")
            
            self.f.write("Stalk LC: " + str(stalklc) +"\n")
            self.f.write("Average Height of Stalks: " + str(avgheight) + "\n")
            self.f.write("Variance in Height of Stalks: " + str(heightvar) + "\n")
            
            self.f.write("max_height: " + str(max_height) +"\nmin_height: " + str(min_height) + "\n")
            self.f.write("Detached Cells: " + str(num_defectors) + "\n")

            
            self.f.write("CLUSTER DATA:\n Total Clusters: " + str(clusters) +"\n")
            self.f.write("Cluster Composition:"+ "\n")
            for cluster in cluster_compositions:
                self.f.write(f"Cluster {cluster['cluster_id']} - Leader Cells: {cluster['leader_cells']}, Follower Cells: {cluster['follower_cells']}, Total Cells: {cluster['total_cells']}\n")  
          
            self.f.write("END")
            
            self.f.close()
        
    def finish(self):     
        if self.output_dir is not None:
            output_path = Path(self.output_dir).joinpath("AreaPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".csv")
            output_path2 = Path(self.output_dir).joinpath("DefPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".csv")
            output_path4 = Path(self.output_dir).joinpath("InfiltratingArea_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".csv")
            output_path5 = Path(self.output_dir).joinpath("OuterBoundary_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".csv")
            output_path6 = Path(self.output_dir).joinpath("MainTumor_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".csv")
            
            self.plot_win1.save_plot_as_data(output_path, CSV_FORMAT)
            self.plot_win2.save_plot_as_data(output_path2, CSV_FORMAT)
            self.plot_win4.save_plot_as_data(output_path4, CSV_FORMAT)
            self.plot_win5.save_plot_as_data(output_path5, CSV_FORMAT)
            self.plot_win6.save_plot_as_data(output_path6, CSV_FORMAT)
            
            png_output_path = Path(self.output_dir).joinpath("AreaPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")
            png_output_path2 = Path(self.output_dir).joinpath("DefPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".png")
            png_output_path4 = Path(self.output_dir).joinpath("InfiltratingArea_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")
            png_output_path5 = Path(self.output_dir).joinpath("OuterBoundary_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".png")
            png_output_path6 = Path(self.output_dir).joinpath("MainTumor_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".png")
            
            self.plot_win1.save_plot_as_png(png_output_path, 1000, 1000 )
            self.plot_win2.save_plot_as_png(png_output_path2, 1000, 1000)
            self.plot_win4.save_plot_as_png(png_output_path4, 1000, 1000)
            self.plot_win5.save_plot_as_png(png_output_path5, 1000, 1000)
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

    