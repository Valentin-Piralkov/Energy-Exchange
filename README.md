# Energy-Exchange
A real world simulation for energy exchange in smart grids.

## Introduction

This project is a simulation of energy exchange in a smart grid. The simulation is based on the paper "Probabilistic Uncertainty in Cooperative
Energy Exchange" by Valentin Piralkov. The simulation is implemented in Python and uses the PuLP library for linear programming.

## Installation

To install the required libraries, run the following command:

```bash pip install numpy``` \
```bash pip install pulp``` \
```bash pip install pandas``` \
```bash pip install matplotlib``` 

## Run the simulation

To run the simulation, run the following command:

```bash python main.py```

## Results

All results are saved in the Plots folder. The results are saved in the form of graphs and tables.

## Data

The data used in the simulation comes from the University of Reading Research Data Archive (https://researchdata.reading.ac.uk/191/).\
The data can be found in the Data folder.

## Environment Variables

All environment variables can be adjusted in the Env.py file.

## Content

The project consists of the following files:

1. main.py: The main file that runs the simulation.
2. Community_Model.py: the definition of a coalition model for energy exchange
3. Individual_Model.py: represents a model where agent operate individually, without energy exchange
4. Community_Utils.py: utility functions for the community model
5. Large_Community.py: a model for a large community (running it requires a lot of computational time!)
6. Probability_Distribution.py: model that predicts future renewable generation using probabilistic distribution
7. Shapley.py: model that calculates the Shapley value for each agent
8. Env.py: the global variables for the simulation

