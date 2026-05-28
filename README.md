# Collective Cancer Invasion: Leader–Follower Dynamics using CompuCell3D

This repository contains the simulation framework, analysis pipeline, processed datasets, visualization resources, and supplementary materials for studying biophysical regulation of leader–follower heterogeneity in collective tumor invasion using the Cellular Potts Model (CPM) implemented in CompuCell3D.

The project accompanies the manuscript:

> **Clusters, Fingers, and Singles: A Mechanical Landscape of Tumor Invasion**

---

# Project Overview

Tumor invasion is a mechanically and biologically heterogeneous process involving coordinated interactions between highly motile leader cells and proliferative follower cells.

This project develops a large-scale biophysical simulation framework using the Cellular Potts Model to:

- model collective cancer invasion,
- quantify invasion morphology,
- identify emergent invasion phenotypes,
- characterize detached cluster dynamics,
- map invasion behaviors across a high-dimensional parameter space,
- and analyze the relationship between adhesion, migration, and proliferation.

The framework combines:

- large-scale CPM simulations,
- automated phenotype classification,
- cluster tracking,
- post-processing analytics,
- parameter-space visualization,
- and interactive supplementary interfaces.

---

# Interactive Supplementary Resources

## Interactive Cluster Trajectory Explorer

An interactive web-based supplementary interface is available for exploring detached cluster centroid trajectories across parameter combinations.

### Launch Explorer

https://sheriffacode.github.io/Cluster_Trajectory_Explorer/

### Explorer Repository

https://github.com/SheriffACode/Cluster_Trajectory_Explorer

### Explorer Features

The interface allows users to interactively explore:

- Leader fraction
- Leader–Follower contact energy (`Jlf`)
- Migration coefficient (`μ`)
- Proliferative probability (`PP`)
- Replicate number
- Simulation time (MCS)
- Cluster trajectories
- Cluster persistence
- Cluster displacement
- Cluster size dynamics

The explorer complements the cluster tracking analysis and supplementary figures associated with the manuscript.

---

# Biological Motivation

Collective invasion is increasingly recognized as a dominant metastatic strategy in many carcinomas, including:

- breast cancer,
- lung cancer,
- colorectal cancer,
- and squamous cell carcinoma.

Experimental studies have demonstrated that tumors frequently contain:

- highly migratory leader cells,
- cohesive follower cells,
- detached multicellular clusters,
- and mixed invasion architectures.

This repository investigates how simple mechanical rules can generate these complex invasion behaviors.

---

# Core Features of the Model

The simulation framework includes:

- Explicit leader–follower heterogeneity
- Chemotaxis-driven leader migration
- Follower-specific proliferation
- Tunable adhesion energies
- Dynamic cluster formation
- Finger-like protrusion detection
- Detached cluster tracking
- Phenotype classification
- Large-scale parameter sweeps
- Automated post-processing analysis

---

# Computational Framework

## Simulation Engine

The model is implemented using:

- CompuCell3D v4.6.0
- Cellular Potts Model (CPM)
- Python steppables
- XML-based CPM configuration

---

# Model Description

The simulation represents tumor invasion within a quasi-two-dimensional spatial domain.

## Cell Types

### Leader Cells (LC)

Leader cells exhibit:

- directional migration,
- chemotaxis,
- invasive protrusion formation,
- reduced proliferation.

### Follower Cells (FC)

Follower cells exhibit:

- high proliferation,
- strong cohesion,
- passive collective movement,
- mechanical coupling to leaders.

---

# Simulated Biophysical Processes

The model incorporates:

- Cell–cell adhesion
- Volume conservation
- Directed migration
- Stochastic motility
- Tumor growth
- Cluster detachment
- Collective invasion
- Single-cell dissemination

---

# Simulation Domain

| Feature | Value |
|---|---|
| Spatial domain | 500 × 300 × 1 |
| Boundary conditions | Periodic in x |
| Geometry | Quasi-2D slab |
| Simulation duration | 700 MCS |
| Cell types | LC, FC |
| Initial configuration | Rectangular tumor slab |

---

# Parameter Space

The project systematically explores:

| Parameter | Symbol | Range |
|---|---|---|
| Leader–Follower Contact Energy | `Jlf` | -5 to 5 |
| Migration Coefficient | `μ` | 0 to 30 |
| Proliferative Probability | `PP` | 0.0 to 1.0 |

Total parameter combinations:

```text
11 × 11 × 11 = 1331
```

Replicates per condition:

```text
10
```

Total simulations:

```text
13310
```

---

# Emergent Invasion Phenotypes

The simulations produce four major invasion phenotypes:

| Phenotype | Description |
|---|---|
| No Invasion | Compact tumor growth |
| Single-Cell Invasion | Solitary cell dissemination |
| Bulk Invasion | Cohesive strand-like invasion |
| Multimodal Invasion | Coexistence of fingers, clusters, and singles |

---

# Installation

## 1. Install CompuCell3D

This project requires:

```text
CompuCell3D v4.6.0
```

Official download:

https://compucell3d.org/SrcBin

---

## Conda Installation

```bash
conda create -n cc3d-env -c compucell3d -c conda-forge compucell3d=4.6.0
conda activate cc3d-env
```

---

# Clone Repository

```bash
git clone https://github.com/Jiang-Lab/Leader_Follower_Invasion_Model.git
cd Leader_Follower_Invasion_Model
```

---

# Running the Simulation

The model uses multiple Python steppables defined in:

```text
CCIecmSteppables.py
```

## Launch Simulation

```bash
cc3d-run -i main_simulation_scan.cc3d
```

---

# Example Steppable Registration

```python
from cc3d import CompuCellSetup
from CCIecmSteppables import (
    NeighborTrackerPrinterSteppable,
    ConstraintInitializerSteppable,
    GrowthSteppable,
    MitosisSteppable
)

CompuCellSetup.register_steppable(
    NeighborTrackerPrinterSteppable(frequency=100)
)

CompuCellSetup.register_steppable(
    ConstraintInitializerSteppable(frequency=1)
)

CompuCellSetup.register_steppable(
    GrowthSteppable(frequency=1)
)

CompuCellSetup.register_steppable(
    MitosisSteppable(frequency=1)
)

CompuCellSetup.run()
```

---

# Repository Structure

```text
.
├── Demo/
├── Data/
├── Figures/
├── Implementation/
├── Plots/
├── Sample/
├── Supp_Videos/
├── Notebooks/
├── Main_Simulation.zip
├── Main_Simulation_Scan.zip
└── README.md
```

---

# Directory Description

## `Demo/`

Contains:

- example simulations,
- demonstration runs,
- representative invasion phenotypes.

---

## `Data/`

Contains processed outputs including:

- invasion metrics,
- phenotype summaries,
- cluster statistics,
- trajectory data,
- extracted boundaries.

---

## `Figures/`

Contains:

- manuscript figures,
- supplementary figures,
- parameter-space plots,
- phenotype maps,
- cluster visualizations.

---

## `Implementation/`

Core implementation files:

- CPM XML configuration,
- Python steppables,
- simulation logic,
- parameter scanning utilities.

---

## `Plots/`

Contains generated visualization outputs including:

- heatmaps,
- volumetric phase maps,
- cluster displacement plots,
- sensitivity analyses,
- trajectory visualizations.

---

## `Sample/`

Representative example outputs and datasets.

---

## `Supp_Videos/`

Supplementary simulation videos associated with the manuscript and supporting information.

---

## `Notebooks/`

Post-processing and analysis notebooks including:

- cluster tracking,
- phenotype classification,
- invasion metric extraction,
- visualization generation,
- statistical analysis.

Example notebooks include:

- `ClusterAnalysisPlots.ipynb`
- `ClusterAnalysisPlots1.ipynb`
- `SimulationDataExtraction.ipynb`
- `Phenotype_composite.ipynb`
- `Phenotypes.ipynb`
- `model_plots.ipynb`
- `invasion_metrics.ipynb`

---

# Output Files

The simulation pipeline produces:

| File | Description |
|---|---|
| `Metrics_Data_*.csv` | Invasion metrics |
| `ClusterComposition_*.csv` | Detached cluster composition |
| `BoundaryData_*.csv` | Tumor boundary coordinates |
| `TumorLeaderCells_*.csv` | Leader spatial coordinates |
| `TumorFollowerCells_*.csv` | Follower spatial coordinates |
| `.png` | Generated plots and figures |

---

# Cluster Tracking

Detached clusters are tracked using:

- centroid-based nearest-neighbor matching,
- persistent TrackIDs,
- splitting/merging detection,
- centroid displacement analysis,
- persistence quantification.

The tracking framework enables:

- trajectory reconstruction,
- persistence analysis,
- cluster migration quantification,
- cluster survival analysis.

---

# Analysis Pipeline

The analysis framework includes:

- BFS-based tumor segmentation
- Convex hull infiltrative area estimation
- Finger detection
- Detached cluster identification
- Single-cell detection
- Phenotype classification
- Random forest sensitivity analysis
- Distance-correlation analysis
- 3D parameter-space visualization

---

# Statistical Analysis

The project uses:

- Pearson correlation
- Spearman correlation
- Distance correlation
- Random forest feature importance
- Parameter-space interpolation
- Replicate averaging

---

# Main Findings

Key findings include:

- Migration and adhesion dominate invasion behavior
- Proliferation primarily affects tumor bulk
- Multimodal invasion emerges robustly
- Detached clusters arise within narrow adhesion–motility windows
- Strong adhesion suppresses dissemination
- Weak adhesion promotes single-cell escape

---

# Supplementary Materials

The repository accompanies extensive supplementary resources including:

- supplementary figures,
- supplementary videos,
- trajectory analyses,
- cluster persistence studies,
- phase-space visualizations,
- 3D simulations,
- parameter sensitivity analyses.

---

# Computational Requirements

Typical runtime:

```text
12–18 minutes per simulation
```

Approximate full sweep cost:

```text
~2800 CPU-hours
```

Parallel execution supported via:

- environment-variable parameterization,
- replicate indexing,
- batch execution.

---

# Citation

If you use this repository, please cite:

```text
Akeeb S, Marcus AI, Jiang Y.
Clusters, Fingers, and Singles:
A Mechanical Landscape of Tumor Invasion.
```

---

# License

MIT License

---

# Contact

## Sheriff Akeeb

Email:

```text
sheriffakeeb@gmail.com
```

---

# Acknowledgements

This work was developed using:

- CompuCell3D
- Python
- NumPy
- SciPy
- Matplotlib
- NetworkX
- GitHub
- GitHub Pages
