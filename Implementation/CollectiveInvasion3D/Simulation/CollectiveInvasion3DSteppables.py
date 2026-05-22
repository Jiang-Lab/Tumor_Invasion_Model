from cc3d.core.PySteppables import *
from datetime import datetime
import numpy as np
import networkx as nx
from pathlib import Path
from scipy.signal import find_peaks
from numpy import trapz
import os
import csv

k = 25
vol = 27
fgrow=0.015 
lgrow = .010 


Jlf = 2
mu = 30 
PP = 0.5

class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)
        self.cellcount_records = []
        
    def start(self):
        
        self.cellcount_filename = os.path.join(self.output_dir, f"CellCount_{Jlf}_{mu}_{PP}.csv")
        self.cellcount_records.append(["MCS", "Leader Cells", "Follower Cells", "Total"])

        #  Target number of leaders based on k%
        total_tumor_cells = len(self.cell_list_by_type(self.LC)) + len(self.cell_list_by_type(self.FC))
        target_num_leaders = int((k / 100) * total_tumor_cells)
        
        while len(self.cell_list_by_type(self.LC)) < target_num_leaders:
            x = np.random.randint(1, self.dim.x - 1)
            y = np.random.randint(1, 20)
            z = np.random.randint(1, self.dim.z)
            c1 = self.cellField[x, y, z]
            
            if c1 and c1.type == self.FC:
                lc = self.new_cell(self.LC)
                self.cellField[x, y, z] = lc
     
        for cell in self.cell_list:
            cell.targetVolume = vol
            cell.lambdaVolume = 2.0
            
         # Initialize gradient field
        g = 1
        mv = self.field.MV
        for x in range(self.dim.x):
            for y in range(self.dim.y):
                for z in range(self.dim.z):
                    mv[x, y, z] = y / g
            
        self.get_xml_element("J_LF").cdata = Jlf
        self.get_xml_element("lambda_chem").Lambda = mu
        
    def step(self, mcs):
        if mcs % 100 == 0:

            Leaders = len(self.cell_list_by_type(self.LC))
            Followers = len(self.cell_list_by_type(self.FC))
            Total = len(self.cell_list)

            self.cellcount_records.append([mcs, Leaders, Followers, Total])

            with open(self.cellcount_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.cellcount_records)
    
class GrowthSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def step(self, mcs):
    
        for cell in self.cell_list_by_type(self.FC):
            cell.targetVolume += fgrow        
    

        
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)

    def step(self, mcs):

        cells_to_divide=[]
        for cell in self.cell_list:
            if cell.volume>2 * vol:
                cells_to_divide.append(cell)

        for cell in cells_to_divide:

            self.divide_cell_random_orientation(cell)
    

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0                  

        self.clone_parent_2_child()            

        
        if self.parent_cell.type==1:
            self.child_cell.type=2
        else:
            self.child_cell.type=1

        