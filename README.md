# Grid-connected-hydrogen-production

## Introduction

This repo records how to build the grid-connected hydrogen production opyimization model.

It is also used for saving all the codes that are used to get the analyzed results of paper:
"Assessing emission certification schemes for grid-connected hydrogen in Australia" in arXiv.

## Installation

You can clone the project directly and then create venv and install dependencies packages by doing following:

```sh
  pip install -r requirements.txt
```

## Solver

The linear model is built using pyomo so any solvers which are compatile with pyomo could be used, such as

- gurobi (very fast and commonly used)
- Cplex
- copt
- CBC  (slow)
- GLPK (slow)
