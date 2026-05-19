

This repository was developed by **Ali Rajaei** for the paper:

> **[Transferable Graph Learning for Transmission Congestion Management via Busbar Splitting]**  
> Published in: [PSCC 2026]  
> Paper link: [[LINK](https://arxiv.org/abs/2510.20591)]

---

## Overview

This repository contains the implementation of:

- DC optimal substation switching (DC-OSS)
- AC optimal substation switching (AC-OSS)
- Congestion-management formulations
- Machine-learning-assisted topology control
- Feedforward neural networks (FNNs)
- Homogeneous graph neural networks (HomoGNNs)
- Heterogeneous graph neural networks (HeteroGNNs)
- Dataset generation pipelines
- Transfer learning and generalization studies
- Computational-efficiency case studies

The framework is developed for power-system topology-control research using:
- Pyomo
- Gurobi
- PyTorch
- PyTorch Geometric

---

## Abstract

[PASTE PAPER ABSTRACT HERE]

---

## Repository Structure

```text
├── DC_Optimal_Splitting_Functions_Pyomo.py
├── AC_Optimal_Splitting_Functions_Pyomo.py
├── DataGeneration_OSS.py
├── Prepare_Features.py
├── FNN_Functions.py
├── HomoGNN_Functions.py
├── HeteroGNN_Functions.py
├── CaseStudies/
├── Datasets/
├── Models/
└── README.md
```

---

## Main Features

### Optimization Framework
- DC-OPF and AC-OPF
- Optimal substation switching (OSS)
- Congestion-management formulations
- Contingency analysis
- Busbar splitting

### Machine Learning Models
- Feedforward neural networks (FNN)
- Homogeneous GNNs
- Heterogeneous GNNs
- Edge-aware graph attention models

### Research Studies
- Prediction accuracy
- Generalization to topology changes
- Cross-system transferability
- Transfer learning
- Computational efficiency

---

## Installation

### Clone repository

```bash
git clone https://github.com/<username>/GNN-OSS-Topology-Control.git
cd GNN-OSS-Topology-Control
```

### Create environment

```bash
conda create -n gnn-oss python=3.12
conda activate gnn-oss
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Main Dependencies

- Python 3.12
- PyTorch 2.6
- PyTorch Geometric
- Pyomo
- Gurobi 12.0
- NumPy
- Pandas
- NetworkX
- Matplotlib

---

## Running Examples

### Generate training data

```bash
python Generate_Training_Data.py
```

### Run prediction-accuracy case study

```bash
python CaseStudy_IVB_PredictionAccuracy.py
```

### Run topology-generalization case study

```bash
python CaseStudy_IVC_GeneralizationTopology.py
```

### Run transferability case study

```bash
python CaseStudy_IVD_Transferability.py
```

### Run computational-efficiency case study

```bash
python CaseStudy_IVE_ComputationalEfficiency.py
```

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{Rajaei2025,
  author  = {Ali Rajaei},
  title   = {Paper Title},
  journal = {Journal Name},
  year    = {2025}
}
```

---

## License

This project is licensed under the MIT License.

---

## Contact

**Ali Rajaei**  
[Email]  
[Google Scholar / LinkedIn / Website]
