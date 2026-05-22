
from cc3d import CompuCellSetup
from CCI_blobSteppables  import (
    NeighborTrackerPrinterSteppable,
    ConstraintInitializerSteppable,
    GrowthSteppable,
    MitosisSteppable,
)


CompuCellSetup.register_steppable(steppable=NeighborTrackerPrinterSteppable(frequency=10))
CompuCellSetup.register_steppable(steppable=ConstraintInitializerSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=GrowthSteppable(frequency=1))
CompuCellSetup.register_steppable(steppable=MitosisSteppable(frequency=1))

CompuCellSetup.run()
