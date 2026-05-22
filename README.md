
# Collective Cancer Invasion: Leader–Follower Dynamics using CompuCell3D

This repository contains simulation code and data for modeling biophysical regulation of leader–follower heterogeneity in collective tumor invasion, using the Cellular Potts Model framework implemented in CompuCell3D.

---

##  Installation

This model runs on **CompuCell3D (v4.6.0)**.

### 1. Download & Install CompuCell3D

Visit the [CompuCell3D download page]([https://compucell3d.org/Downloads](https://compucell3d.org/SrcBin)) and install for your OS:

- Windows: Installer available
- macOS/Linux: Use conda installation

Or install using `conda`:

```bash
conda create -n cc3d-env -c compucell3d -c conda-forge compucell3d=4.6.0
conda activate cc3d-env
```

### 2. Clone this Repository

```bash
git clone https://github.com/SheriffACode/tumor-invasion-model.git
cd tumor-invasion-model
```

---

##  Running the Simulation

The model uses multiple `Steppables` defined in `CCIecmSteppables.py` and can be launched using:

```python
from cc3d import CompuCellSetup
from CCIecmSteppables import (
    NeighborTrackerPrinterSteppable,
    ConstraintInitializerSteppable,
    GrowthSteppable,
    MitosisSteppable
)

CompuCellSetup.register_steppable(NeighborTrackerPrinterSteppable(frequency=100))
CompuCellSetup.register_steppable(ConstraintInitializerSteppable(frequency=1))
CompuCellSetup.register_steppable(GrowthSteppable(frequency=1))
CompuCellSetup.register_steppable(MitosisSteppable(frequency=1))

CompuCellSetup.run()
```

You can run simulations using:

```bash
cc3d-run -i main_simulation_scan.cc3d
```
where 'main_simulation_scan.cc3d' links the XML model and the python code
---

## Parameter Space

```python
parameter_values = {
    'Jlf': [-5, -4, ..., 5],
    'mu': [0, 3, ..., 30],
    'PP': [0.0, 0.1, ..., 1.0]
}
```

Each simulation is indexed by `iteration`, used to load the corresponding parameter combination.

---

## Output Data

- `Metrics_Data_*.csv`: Invasive/finger area, clusters, defectors
- `ClusterComposition_*.csv`: Composition of each detached cluster
- `.png`: Boundary and phenotype plots
- `BoundaryData_*.csv` — Tumor boundary coordinates
- `TumorLeaderCells_*.csv`, `TumorFollowerCells_*.csv` — leader/follower Spatial positions
---

## License

MIT License

---

## Contact
For questions and enquiries, please reach out via sheriffakeeb@gmail.com



---

## Model Description

This model simulates **2D tumor invasion dynamics** using a **Cellular Potts Model (CPM)** in CompuCell3D. The simulation captures emergent behaviors driven by **biophysical heterogeneity** and **regulatory mechanisms** between distinct cancer cell subtypes.

### Simulated Features:

- **Leader Cells (LC)** and **Follower Cells (FC)** with distinct behaviors
- **Motility via chemotaxis**: Leader cells move directionally in response to a fixed chemoattractant field
- **Follower-specific proliferation**, governed by a tunable proliferative probability (PP)
- **Cell–cell adhesion regulation**, particularly between LC and FC (`J_lf`)
- **Quantification of invasion** via:
  - Invasive Area
  - Infiltrative Area
  - Finger-like protrusions
  - Detached clusters
  - Tumor front complexity

---

### XML Configuration Highlights

| Feature                     | Setting                                                  |
|----------------------------|-----------------------------------------------------------|
| **Cell Types**             | `Medium`, `LC` (leaders), `FC` (followers)                |
| **Spatial Domain**         | 500 x 300 x 1 (2D) grid                                   |
| **Boundary Conditions**    | Periodic in **x**, fixed in **y**                         |
| **Simulation Duration**    | 701 Monte Carlo Steps (MCS)                              |
| **Chemotaxis**             | Enabled for `LC` toward chemical field `MV`              |
| **Adhesion Energies**      | `Contact` plugin with `J` values fully configurable       |
| **Proliferation**          | Applied to a fraction of FCs based on input PP            |
| **PDE Field**              | Diffusion-free, externally imposed chemoattractant `MV`   |
| **Initial Cell Placement** | Rectangular slab of FCs; LCs randomly assigned by ratio `k%` |


