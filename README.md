# PanasonicGP
A complete GP regression model for predicting curb height in collaboration with Panasonic

RawData
- contain all the data in raw format
- run RawExtract.py to obtain the desirable data format (refer to Data folder)

Data
- contain all the data for GP hyper-parameter learning and prediction
- two class of data:
  1. raw signal
rawTrain.txt: contains N * (D+1) elements where N is the size of training data and D is input dimension (D = 240), first D column is the training input and last column is the training output
rawTest.txt: contains T * (D+1) elements where T is the size of testing data, first D column is the testing input and last column is the test output
  2. features representing the raw signal
comfeat_train(#).txt: contains N * (D+1) elements, the input dimension is much smaller compared to raw signal
comfeat_test(#).txt: contains N * (D+1) elements

Python
- Install GPy:
  1.install anaconda
  2.in command line type:
    conda update scipy
    pip install gpy
    pip install --upgrade GPy
  3.this folder contains the GPy code for Gaussian process learning hypers and prediction
  4.type “python GPy-hyper.py directory” to run the program, directory contains the location where data is stored (in Data folder)
  5.type “python GPy-hyper.py directory sid” to run a particular dataset, sid range from -3 to 3.
- prediction results and hypers are stored in the location specified by “directory”
- prediction results are specified by res[N].txt where N range from -3 to 3. when N is negative, the results are obtained by exploiting features alone; when N = 0, the results are obtained by exploiting raw signal alone; when N > 0 the results are obtained by combining raw signal with features
- corresponding hyper-parameters are stored in res-hyp[N].txt, N range from -3 to 3.
- if the results are not so good, try to run GPy-hyper.py several times

Matlab
1.need to run createData.mat to generate ‘sonarData.mat’. In matlab command line, type createData(‘directory’) where ‘directory’ contains the location of data
2.run rawTrain.m to obtain the hyper-parameters (require ‘sonarData.mat’). In matlab command line, type rawTrain(N) where N range from -3 to 3.
3.run marplot.m to visualize
4.pyplot.m takes in the output of GPy-hyper.py (i.e., res[N].txt) and visualize GPy’s result

Predict
- contain the C code for GP prediction
1.type “make” to compile
2.to run the program, type "./gp train_file_name test_file_name hyper_file_name"
(./gp gpTrain0.txt gpTest0.txt res-hyp0.txt)
3.train and test file has the same format in Data folder, train file contains N * (D+1) elements and test file contains T * (D+1) elements
4.format of hyper file:
- Line1: signal, noise, mean of train output, variance of train output, input_dim, input_num, output_num
- Follow with input_dim lines: each line contains a length_scale
- all the hypers are in log form
