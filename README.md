
[![Paper](http://img.shields.io/badge/paper-arxiv.2402.05309-B31B1B.svg)](https://arxiv.org/abs/2402.05309)
[![Python versions](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

# Investigating Generalization Behaviours of Generative Flow Networks (GFlowNets, GFNs)

This repo is modified and based off the implementation: [gflownet](https://github.com/recursionpharma/gflownet.git) - it contains GFlowNet-related training and environment code on graphs for investigating the generalization capabilities of GFlowNets.

## GFlowNets and generalization

It is hypothesize that GFlowNets leverage the generalization potential of deep neural networks to assign probability mass to unvisited areas of the state space. This repo contains a graph generation benchmark environment (considering several rewards of varying difficulty) such that $p(x)$ can be tractably computed over all graphs. This set of graph generation tasks are used to assess the ability of GFlowNets to *generalize* and approximate $p(x)$ for unvisited areas of the state space. See [this http link](https://arxiv.org/abs/2402.05309) for the paper.

## GFlowNets

[GFlowNet](https://yoshuabengio.org/2022/03/05/generative-flow-networks/), short for Generative Flow Network (sometimes also abbreviated as GFN), is a novel generative modelling framework for learning unnormalized probability mass functions over discrete spaces, particularly suited for discrete/combinatorial objects. Here, the focus is on graph generation.

The idea behind GFlowNets is to estimate flows in a (graph-theoretic) directed acyclic network. The network represents all possible ways of constructing an object, and so knowing the flow gives us a policy which we can follow to sequentially construct objects. Such a sequence of partially constructed objects is a _trajectory_. *Perhaps confusingly, the _network_ in a GFlowNet refers to the state space, not a neural network architecture*. Here the objects we construct are themselves graphs, which are constructed node by node. To make policy predictions, we use a graph neural network, parameterizing the forward policy $P_F(s' | s; \theta)$. This GNN outputs per-node logits (e.g. add a node to the current graph, or add an edge between these two nodes), as well as per-graph logits (e.g. stop/"done constructing this object").

## Citing

If you find this code useful in your research, please cite the following paper (expand for BibTeX):

<details>
<summary>
L. Atanackovic, E. Bengio. Investigating Generalization Behaviours of Generative Flow Networks, 2024.
</summary>

```bibtex
@article{atanackovic2024,
  title={Investigating Generalization Behaviours of Generative Flow Networks},
  author={Atanackovic, Lazar and Bengio, Emmanuel},
  journal={arXiv preprint arXiv:2402.05309},
  year={2024}
}
```
</details>

## Repo overview

Structure of repo:

- [algo](src/gflownet/algo), contains GFlowNet algorithms implementations ([Trajectory Balance](https://arxiv.org/abs/2201.13259), [SubTB](https://arxiv.org/abs/2209.12782), [Flow Matching](https://arxiv.org/abs/2106.04399)), as well as some baselines. These implement how to sample trajectories from a model and compute the loss from trajectories.
- [data](src/gflownet/data), contains dataset definitions, data loading and data sampling utilities.
- [envs](src/gflownet/envs), contains environment classes; a graph-building environment base, and a molecular graph context class. The base environment is agnostic to what kind of graph is being made, and the context class specifies mappings from graphs to objects (e.g. molecules) and torch geometric Data.
- [examples](docs/examples), contains simple example implementations of GFlowNet.
- [models](src/gflownet/models), contains model definitions.
- [tasks](src/gflownet/tasks), contains training code.
    -  [`basic_graph_task.py`](src/gflownet/tasks/basic_graph_task.py), graph generation environment for counting, neighbors, and cliques tasks. 
- [utils](src/gflownet/utils), contains utilities (multiprocessing, metrics, conditioning).
- [`trainer.py`](src/gflownet/trainer.py), defines a general harness for training GFlowNet models.
- [`online_trainer.py`](src/gflownet/online_trainer.py), defines a typical online-GFN training loop.

See [implementation notes](docs/implementation_notes.md) for more.

## Getting started

First, generate a cache of all states/graphs up to 7 nodes. To do this run the following:

```bash
python basic_graph_task.py --recompute-all ./data/basic_graphs 7
```

To train a single model on the graph benchmark tasks, run:

```bash
python expts/task_single_run_gfn.py
```

To run an experiment, e.g. training the distilled flow models for $F$ and $P_F$ over 3 seeds, use [`task_distilled_flows.py`](expts/task_distilled_flows.py). 

## Installation

### PIP

This package is installable as a PIP package, but since it depends on some torch-geometric package wheels, the `--find-links` arguments must be specified as well:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cu117.html
```
Or for CPU use:

```bash
pip install -e . --find-links https://data.pyg.org/whl/torch-1.13.1+cpu.html
```

To install or [depend on](https://matiascodesal.com/blog/how-use-git-repository-pip-dependency/) a specific tag, for example here `v0.0.10`, use the following scheme:
```bash
pip install git+https://github.com/recursionpharma/gflownet.git@v0.0.10 --find-links ...
```

If package dependencies seem not to work, you may need to install the exact frozen versions listed `requirements/`, i.e. `pip install -r requirements/main_3.9.txt`.

