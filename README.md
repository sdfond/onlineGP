# PanasonicGP
A complete GP regression model for predicting curb height in collaboration with Panasonic

RawData
- contain all the data in raw format
- Raw data files include:
  - processed.data.[tid]-NUS_fv[fid].csv, where tid is the time signature and fid is selected from ['A', 'B', 'C'].
  - fv[fid].train.txt, this file contains the SVM results. It also contains test_json_file and ignore-id if applicable.
  - train-NUS.fv[fid].txt, this file specifies how the IDs are combined (each ID represents a sonar profile in csv file)
- run RawExtract.py to obtain the desirable data format (refer to Data folder)
- How to use RawExtact.py:
  - type "python RawExtract.py timeid" where timeid is used to specify the data, for instance, timeid = 20160706 or 20160527
  - after running RawExtract.py, you need to check whether the number of training and testing data is correct (verify it with the output of SVM result)
- How to use multiRawExtract.py
  - this is the version which will apply to multiple subfolders
  - you need to change the subfolder name
  - if the ignore-id exists (refer to train-NUS.fvA.txt, last row), you need to copy the id into removeList (inside the python code)

Data
- contain all the data for GP hyper-parameter learning and prediction
- two class of data:
  1. raw signal
rawTrain.txt: contains N * (D+1) elements where N is the size of training data and D is input dimension (D = 240), first D column is the training input and last column is the training output
rawTest.txt: contains T * (D+1) elements where T is the size of testing data, first D column is the testing input and last column is the test output
  2. features representing the raw signal
comfeat_train(#).txt: contains N * (D+1) elements, the input dimension is much smaller compared to raw signal
comfeat_test(#).txt: contains N * (D+1) elements

Install GPy:
- install anaconda
- in command line type:
  - conda update scipy
  - pip install gpy
  - pip install --upgrade GPy

Python Folder
- this folder contains the GPy code for Gaussian process learning hypers and prediction
- type “python GPy-hyper.py directory opti_method” to run the program (all 7 datasets combination) with optimization method set to opti_method (either bfgs or scg), directory contains the location where data is stored (in Data folder)
- prediction results and hypers are stored in the FOLDER WHERE YOU RUN GPy-hyper.py
- prediction results are specified by res[N].txt where N range from -3 to 3. when N is negative, the results are obtained by exploiting features alone; when N = 0, the results are obtained by exploiting raw signal alone; when N > 0 the results are obtained by combining raw signal with features
- in res[N].txt, the first two lines are the statistical results of classification. Each line contains 2 number [a b], where a is the number of testing samples in this class and b is the number of samples sucessfully classified.
- corresponding hyper-parameters are stored in res-hyp[N].txt, N range from -3 to 3.
- you can also type “python GPy-hyper.py directory opti_method sid” to run a particular dataset, sid range from -3 to 3. For instance, type "python GPy-hyper.py ../Data/5-10/ bfgs 0", it will only train the hyper-parameters of raw signal alone with optimization method bfgs.
- IMPORTANT: if the results are not as expected, you can:
  - run GPy-hyper.py several times
  - change the optimization method from scg to bfgs or vice versa

Matlab
- need to run createData.mat to generate ‘sonarData.mat’. In matlab command line, type createData(‘directory’) where ‘directory’ contains the location of data
- run rawTrain.m to obtain the hyper-parameters (require ‘sonarData.mat’). In matlab command line, type rawTrain(N) where N range from -3 to 3.
- run marplot.m to visualize
- pyplot.m takes in the output of GPy-hyper.py (i.e., res[N].txt) and visualize GPy’s result

Predict
- contain the C code for GP prediction
- type “make” to compile
- to run the program, type "./gp train_file_name test_file_name hyper_file_name". For instance, "./gp gpTrain0.txt gpTest0.txt res-hyp0.txt"
- training and testing file has the same format in Data folder, training file contains N * (D+1) elements and testing file contains T * (D+1) elements
- format of hyper file:
  - Line1: signal, noise, mean of train output, variance of train output, input_dim, input_num, output_num
  - Follow with input_dim lines: each line contains a length_scale
- all the hypers are in log form
