# Autofly Test Bench

This is the Autofly Test Bench where we can collaborate to spare us from having
to build up our own test environments each. The idea is to develop a modular
test bench for controllers, estimators, models, environments, etc., which will
be implemented in Python3.5+. Of course, MATLAB code is also welcome, as there
is probably a notable amount of code that would have to be ported otherwise.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
### Prerequisites

Get Anaconda from their [project webpage](https://www.anaconda.com/distribution/) and install it on your system.

### Installing

Clone the repository and navigate to the directory
```
git clone git@uavlab.itk.ntnu.no:dirkpr/autofly.git
cd autofly
```

Create the python environment and activate it.
```
conda env create -f py35.yml
```

Verify that the environment was installed correctly
```
conda list
```

If needed, you can find more information on managin python environments using conda on their [webpage](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

### Usage
Activate the environment and run the simulation.
```
conda activate py35
python main.py
```
