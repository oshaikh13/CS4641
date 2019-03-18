The Jupyter-Notebooks contained in this .zip file were run using Python 3.6
Analysis and modeling was done with sci-kit learn and a forked version of mlrose.

The Credit Card data is included in the data directory.

To run this on a standard CoC Linux box, first download and setup Anaconda. 

Then, download the following dependencies (all latest versions):
numpy 
pandas
sklearn
matplotlib
imblearn
mlrose -- note that I forked this, so download the version from this URL: https://github.com/oshaikh13/mlrose
          Then, run python setup.py install

Each Jupyter notebook can be run from top to bottom, and the graphs used in the attached
analysis are generated. View the Neural Network notebook to view the NN optimization procedure. 
Similarly, view General Optimization to see that procedure. Respective Plots are generated 
in the notebooks themselves. You can also edit the data directories from the notebooks themselves.

Citations for dependencies:
Hayes, G.mlrose: Machine Learning, Randomized Optimization and SEarch package for Python.
   https://github.com/gkhayes/mlrose. Accessed: 22 February 2019, 2019