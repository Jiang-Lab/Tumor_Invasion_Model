
from cc3d import CompuCellSetup
from CCIecmSteppables import NeighborTrackerPrinterSteppable

CompuCellSetup.register_steppable(steppable=NeighborTrackerPrinterSteppable(frequency=100))
        


from CCIecmSteppables import ConstraintInitializerSteppable

CompuCellSetup.register_steppable(steppable=ConstraintInitializerSteppable(frequency=1))





from CCIecmSteppables import GrowthSteppable

CompuCellSetup.register_steppable(steppable=GrowthSteppable(frequency=1))




from CCIecmSteppables import MitosisSteppable

CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=1))








CompuCellSetup.run()

import os
print("✅ Current working directory:", os.getcwd())
