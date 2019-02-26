The Jupyter-Notebooks contained in this .zip file were run using Python 3.6
Analysis and modeling was done with sci-kit learn and Keras (tensorflow backend).

NOTE: The PCam dataset is far too large to upload as a zip; therefore, only the
Credit Card Fraud Data is included. To run the PCam notebooks, simply download the data zip 
from the following Kaggle URL: https://www.kaggle.com/c/histopathologic-cancer-detection. 
Then, unzip it into the data/PCam directory. Ensure that you have also unzipped the 
train and test subdirectories inside the PCam archive.

To run this on a standard CoC Linux box, first download and setup Anaconda. 

Then, download the following dependencies (all latest versions):
numpy 
pandas
sklearn
matplotlib
tensorflow
imblearn
keras

Each Jupyter notebook can be run from top to bottom, and the graphs used in the attached
analysis are generated. View the PCam CNN notebook's last few cells to see the comparative
analysis visualizations. You can also edit the data directories from the notebooks themselves.

Citations for dependencies:
Scikit-Learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
Keras. Chollet, Francios et al. https://keras.io. 2015