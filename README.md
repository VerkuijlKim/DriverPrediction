# Forensic Driver Prediction using ML and LR

## Getting started
To run the code, you will need the following dependencies installed:
* numpy
* pandas
* seaborn
* matplotlib
* sklearn
* scipy
* itertools
* lir (https://github.com/NetherlandsForensicInstitute/lir)
* torch (only for running the NN)

## Files
### implementation.py
This model compares our main 3 models in the One-vs-One comparison situation, where one driver is in-distribution and the second driver is out-of-distribution. This file can show the performance metrics, by reading the stored test results from a csv.

### train_test.py
To get the csv file used in implementation.py, you can run this file.

### ood_models.py
Contains the One-class SVM, Local Outlier Factor and Isolation Forest Model.

### Functions(2).py and DataDefined(2).py
Utility functions

###

