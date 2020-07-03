# MARBLES-CHONK-NAME_TO_FIND

Flexible Semi-Lagrangian Explicit Landscape Evolution Model with Conditionless Depressions Solver and Provenance Tracking in Heterogeneous Environment. FSLELEMCDSPTHE not being a very nice name, we are currently trying to find one.

TODO add short description here

This model will be part of the fastscape ecosystem. It is written in full combination of C++, Fortran and python. Python manage the model run and communicate with fastscalib-fortran to get node-ordering, then send everything to c++.


## Installation

### (optional) Using conda to manage dependencies

TODO explain briefly how to use conda

### Installing the python package

 - clone this repository
 - `pip install .`


## Main Features

- Ensure comprehensive topological order from top to bottom with an unaltered topography and communicating depressions.

## Code Structure

### Node graph

The node graph is approaching its final form. The object, stored in `nodegraph.c/hpp`, provides tool to ingest information from `fastscapelib-fortran` and preprocess them for the model. The main difference (and very painful point to work on) from `fastscapelib-fortran` is that the depressions' topography is not altered with carving, it is instead left intact. However it considers that the depressions needs to be able to overflow in a meaningful way, it therefore needs to be processed in the right order. The preprocessing uses Cordonnier et al., 2019 to simulate a fake receiver to each pit nodes and rerun a topological sort on the obtained node graph. The topological order hence obtain guaranties that every nodes in the graph will be processed after all of his donors. If a depression overflows, it will be before it's receiving counterparts, if not, well it does not matter but at least it will have received all potential water available. The particular case of imbricated depressions will be treated using a "connected vessel" algorithm later develloped.
