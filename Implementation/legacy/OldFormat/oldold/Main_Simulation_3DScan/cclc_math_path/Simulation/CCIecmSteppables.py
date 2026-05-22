from cc3d.cpp.PlayerPython import * 
from cc3d import CompuCellSetup

from cc3d.core.PySteppables import *
from datetime import datetime
import numpy as np
import networkx as nx
from scipy.signal import argrelextrema
from pathlib import Path



k = 25 #(5) <1-10,10-80> This is the percent concentration of Leaders in the tumor
matrix = np.zeros((300,500), int)
lccelldiv=0
fccelldiv=0
divtime=0
fgrow=0.015 #(.11) <.005 to .015> growth rate for Followers 
lgrow = .010 #(.015) growth rate for Leaders (less than Followers)


Jlf = {{Jlf}} #The Adhesion (contact energy) of leader cells and follower cells
mu = {{mu}} #Chemotaxis Lambda
PP = {{PP}} # the percentage of the follower cells allowed to proliferate

class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
       
        
        SteppableBasePy.__init__(self,frequency)
        


    def start(self):
        
        #try: 
        global tracklc, k
        self.m = open(self.output_dir +"\CellCount_"+ str(Jlf)+"_"+str(mu)+"_"+str(PP)+".txt", "a")
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
        
        
        self.plot_win4 = self.add_new_plot_window(title='Infiltrative Area Over Time',
                                                 x_axis_title='MonteCarlo Step (MCS)',
                                                 y_axis_title='Area', 
                                                 x_scale_type='linear', 
                                                 y_scale_type='linear',
                                                 grid=False)
        
        self.plot_win4.add_plot("AUC", style='Lines', color='red', size=2)
        
        
        self.plot_win5 = self.add_new_plot_window(title='Outer Boundary Shape',
                                                 x_axis_title='x area covered',
                                                 y_axis_title='y area covered', 
                                                 x_scale_type='linear', 
                                                 y_scale_type='linear',
                                                 grid=False)
        
        self.plot_win5.add_plot("Outer_Boundary", style='Dots', color='red', size=2)
        # Add plot for the smooth curve joining the outermost points
        self.plot_win5.add_plot("Outer_Boundary_Curve", style='Lines', color='blue', size=2)
        
        
        
        
        
        
        
    def step(self, mcs): 
        
        #Measure the infiltrative area using the AUC of the outer boundary formed by the topmost cell in each column of the simulation domain.
        if mcs % 100 == 0:
            xmax, ymax = 500, 300
            outer_boundary = [] # Store the outer boundary coordinates

            # Iterate through columns to find the topmost non-medium cell
            for x in range(xmax):
                for y in range(ymax - 1, -1, -1):  # Scan from top to bottom
                    cell = self.cell_field[x, y, 0]
                    if cell and cell.type != self.MEDIUM:
                        outer_boundary.append((x, y))
                        break

            if outer_boundary:
                # Sort boundary points by x-coordinate
                outer_boundary = sorted(outer_boundary, key=lambda p: p[0])
                x_coords, y_coords = zip(*outer_boundary)
                # Plot the raw outer boundary points
                for x, y in zip(x_coords, y_coords):
                    self.plot_win5.add_data_point("Outer_Boundary", x, y)
                

                # Identify the upper boundary points
                upper_boundary = {}
                for x, y in outer_boundary:
                    if x not in upper_boundary or y > upper_boundary[x]:
                        upper_boundary[x] = y
                upper_x = sorted(upper_boundary.keys())
                upper_y = [upper_boundary[x] for x in upper_x]

                # Calculate the infiltrative area (AUC) using the trapezoidal rule               
                auc = np.trapz(y_coords, x_coords) 
            else:
                auc = 0
                
            
                
               
            self.plot_win4.add_data_point("AUC", mcs, auc)

    
            self.m.write("Step: " +str(mcs)+ "\n")
            self.m.write("LC:" + str(len(self.cell_list_by_type(self.LC))) + "\n")
            self.m.write("FC:" + str(len(self.cell_list_by_type(self.FC))) + "\n")
            self.m.write("Infiltrative Area (AUC):" + str(auc) + "\n")
            
        
        if mcs == 700:
            if outer_boundary:
                # Plot the raw outer boundary points
                #for x, y in zip(x_coords, y_coords):
                    #self.plot_win5.add_data_point("Outer_Boundary", x, y)
                # Plot the upper boundary curve
                for x, y in zip(upper_x, upper_y):
                    self.plot_win5.add_data_point("Outer_Boundary_Curve", x, y)
            
            
            self.m.write("Final number of Cells: " +"\n")
            self.m.write("Final LC: " + str(len(self.cell_list_by_type(self.LC))) + "\n")
            self.m.write("Final FC: " + str(len(self.cell_list_by_type(self.FC))) + "\n")
            
            
            if self.output_dir is not None:
                output_path4 = Path(self.output_dir).joinpath("InfiltratingArea_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".csv")
                output_path5 = Path(self.output_dir).joinpath("OuterBoundary_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".csv")
                self.plot_win4.save_plot_as_data(output_path4, CSV_FORMAT)
                self.plot_win5.save_plot_as_data(output_path5, CSV_FORMAT)

                png_output_path4 = Path(self.output_dir).joinpath("InfiltratingArea_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")
                png_output_path5 = Path(self.output_dir).joinpath("OuterBoundary_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")
                self.plot_win4.save_plot_as_png(png_output_path4, 1000, 1000)
                self.plot_win5.save_plot_as_png(png_output_path5, 1000, 1000)
            
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
        self.f = open(self.output_dir +"\data_" +  str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".txt", "a")
        
        self.plot_win = self.add_new_plot_window(
            title='Invasive Area Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Invasive AREA',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False # only in 3.7.6 or higher
        )
        self.plot_win2 = self.add_new_plot_window(
            title='Defectors Over Time: ' + str(Jlf)+"_"+str(mu)+"_"+str(PP),
            x_axis_title='MCS',
            y_axis_title='Number of Defectors',
            x_scale_type='linear',
            y_scale_type='linear',
            grid=False # only in 3.7.6 or higher
        )
        '''
        # initialize setting for Histogram
        self.plot_win3 = self.add_new_plot_window(
            title='Histogram of Composition of Cluster Cells',
            x_axis_title='Cluster size',
            y_axis_title='Frequency'
        )
        
        self.plot_win3.add_histogram_plot(plot_name='Cluster Composition', color='green', alpha=100)
        '''

        self.plot_win.add_plot("Invasive AREA", style='Lines', color='red', size=2)
        self.plot_win2.add_plot("defectors", style='Dots', color='red', size=5)
        self.plot_win2.add_plot("defectors+clusters", style='Lines', color='blue', size=2)
        
        
        self.create_scalar_field_cell_level_py("myField")  

    def step(self, mcs):
        global max
        tips = self.field.myField
        tips.clear()

        if mcs == 0:
            max = 40

        queue = []
        tumorcells = []
        stalkcells = []
        min = 30
        surface = []
        Tumor = nx.Graph()
        Tumor.clear()

        def bfs(visited, node):  # function for BFS
            visited.append(node.id)
            queue.append(node.id)
            Tumor.add_node(node.id, xCOM=node.xCOM, yCOM=node.yCOM)  # Store real COM values

            while queue:  # Creating loop to visit each node
                mID = queue.pop(0)
                m = self.fetch_cell_by_id(mID)  # Fetch the actual cell object by ID
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

        defectorcells = []
        for cell in self.cell_list_by_type(self.LC):
            n = 0
            for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                if neighbor:
                    n += 1
            if n == 0:
                defectorcells.append(cell.id)

        for x, y, z in self.every_pixel():
            cell = self.cell_field[x, y, z]
            if not cell and y < min:
                min = y

        self.f.write("Step" + str(mcs) + "\n")
        self.f.write("LC:" + str(len(self.cell_list_by_type(self.LC))) + "\n")
        self.f.write("MIN:" + str(min) + "\n")

        # Track the cluster composition
        cluster_compositions = []

        tumorcells = []  # List for visited nodes.
        clustercells = []
        queue = []  # Initialize a queue
        clusters = 0
        stalklc = 0

        for x in range(0, 499):
            cell0 = self.cell_field[x, 1, 0]
            if cell0:
                if cell0.id not in tumorcells:
                    tumorcells += bfs(tumorcells, cell0)

        for cell in self.cell_list_by_type(self.LC):
            if cell.id in tumorcells:
                stalklc += 1
                stalkcells.append(cell)

        for cell in self.cell_list_by_type(self.FC):
            if cell.id not in tumorcells:
                queue.clear()
                clusters += 1  # Increment cluster count
                clustercells = bfs([], cell)
                tumorcells.extend(clustercells)

                # Now count the number of leader and follower cells properly
                num_leader_cells = sum(1 for cell_id in clustercells if self.fetch_cell_by_id(cell_id).type == self.LC)  # Leader cells
                num_follower_cells = sum(1 for cell_id in clustercells if self.fetch_cell_by_id(cell_id).type == self.FC)  # Follower cells

                # Store the cluster composition
                cluster_compositions.append({
                    "cluster_id": clusters,
                    "leader_cells": num_leader_cells,
                    "follower_cells": num_follower_cells,
                    "total_cells": len(clustercells)
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

        vsurface = []
        queue = []
        vsurface.append(startcell)
        queue.append(startcell)
        endpoints = []
        epnodes = []
        epheight = []
        for i in list(Surface.nodes):
            cell = self.fetch_cell_by_id(i)
            if Surface.degree(i) < 5:
                endpoints.append(cell)
                epnodes.append(i)
                tips[cell] = 100

        for c in list(endpoints):
            epheight.append(c.yCOM - min)
        stalks = len(endpoints)
        epheightavg = np.mean(epheight)

        for id in Surface.nodes:
            cell = self.fetch_cell_by_id(id)
            if cell and cell.yCOM < min:
                min = cell.yCOM
        for id in Tumor.nodes:
            cell = self.fetch_cell_by_id(id)
            if cell:
                if cell.id not in clustercells:
                    if cell.id not in defectorcells:
                        invarea += cell.volume

        for cell in self.cell_list_by_type(self.LC):
            if cell.id in clustercells:
                clusterlc.append(cell)
        for cell in self.cell_list_by_type(self.FC):
            if cell.id in clustercells:
                clusterfc += 1

        invarea = invarea - (min * 500)
        avgheight = 0
        heightvar = 0
        if stalks > 0:
            avgheight = np.mean(epheight)
            heightvar = np.var(epheight)
            max = np.amax(epheight)
        self.f.write("stalks: " + str(stalks) + "\n")
        self.f.write("Average Height of Stalks: " + str(avgheight) + "\n")
        self.plot_win.add_data_point('Invasive AREA', mcs, invarea)

        defects = len(defectorcells) + len(clusterlc) + clusterfc
        self.plot_win2.add_data_point('defectors+clusters', mcs, defects)
        self.f.write("Variance in Height of Stalks: " + str(heightvar) + "\n")
        self.f.write("Invasive Area: " + str(invarea) + "\n")
        
        total_cells_array = []
        
        # Log cluster composition at each time step
        '''
        self.f.write(f"Step {mcs}: Cluster Composition\n")
        for cluster in cluster_compositions:
            total_cells_array.append(cluster['total_cells'])
            self.f.write(f"Cluster {cluster['cluster_id']} - Leader Cells: {cluster['leader_cells']}, Follower Cells: {cluster['follower_cells']}, Total Cells: {cluster['total_cells']}\n")
        #self.plot_win3.add_histogram(plot_name='Cluster Composition', value_array=total_cells_array, number_of_bins=5)
        ''' 
        
        if mcs == 700:
            
            
            
            perimeter = 0
            for u,v,e in Surface.edges(data=True):
                perimeter += e['weight']
            complexity = np.square(perimeter)/(4*np.pi*invarea)
            self.f.write("Tumor perimeter: " + str(perimeter) + "\n")
            self.f.write("Tumor complexity: " + str(complexity) + "\n")         
            self.f.write("Tumor endpoints: " + str(len(endpoints)) + "\n")
            self.f.write("Tumor cells: " + str(len(list(Tumor.nodes))) + "\n")
            
            self.f.write("Invasive Area: " + str(invarea) +"\n")
            self.f.write("Defected Leaders: " + str(len(defectorcells)) + "\n")
            self.f.write("Stalks: " + str(stalks) + "\n")
            self.f.write("Stalk Leaders: " + str(stalklc) +"\n")
            self.f.write("Variance in Height of Stalks: " + str(heightvar) + "\n")
            self.f.write("MAX: " + str(max) +"\nMIN: " + str(min) + "\n")
            
            self.f.write("CLUSTER DATA:\n Total Clusters: " + str(clusters) +"\n")
      
            
            
            self.f.write("Cluster Composition:"+ "\n")
            for cluster in cluster_compositions:
                total_cells_array.append(cluster['total_cells'])
                self.f.write(f"Cluster {cluster['cluster_id']} - Leader Cells: {cluster['leader_cells']}, Follower Cells: {cluster['follower_cells']}, Total Cells: {cluster['total_cells']}\n")

            self.f.write("END")
            if self.output_dir is not None:
                output_path = Path(self.output_dir).joinpath("AreaPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".csv")
                output_path2 = Path(self.output_dir).joinpath("DefPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+  ".csv")
                self.plot_win.save_plot_as_data(output_path, CSV_FORMAT)
                self.plot_win2.save_plot_as_data(output_path2, CSV_FORMAT)

                png_output_path = Path(self.output_dir).joinpath("AreaPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")
                png_output_path2 = Path(self.output_dir).joinpath("DefPlots_" + str(Jlf)+"_"+str(mu)+"_"+str(PP)+ ".png")

                # here we specify size of the image saved - default is 400 x 400
                self.plot_win.save_plot_as_png(png_output_path, 1000, 1000)
                self.plot_win2.save_plot_as_png(png_output_path2, 1000, 1000)
            self.f.close()
     
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self, frequency=1):
        MitosisSteppableBase.__init__(self, frequency)
        self.cell_to_proliferate = []

    def start(self):
        '''
        Initialize the clock for 50% of the Follower Cells (FC).
        If the cell is selected to proliferate (50% proliferative probability), assign it a clock.
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

    