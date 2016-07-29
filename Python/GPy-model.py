# Gpy-hyper.py takes in 2 input parameters: working directory and data indication.
# The second parameter is optional, if not specified, will run all the possible data combinations.
import GPy as gp
import numpy as np
import sys
import os

# display the directory and change to that directory
cur_d = os.getcwd()
print 'change to directory:', sys.argv[1]
os.chdir(sys.argv[1])
# load in training and testing data
train = np.loadtxt('rawTrain.txt')
test = np.loadtxt('rawTest.txt')


feat1_train = np.loadtxt('comfeat_train0.txt')
feat2_train = np.loadtxt('comfeat_train1.txt')
feat3_train = np.loadtxt('comfeat_train2.txt')
trainFeat = [feat1_train,feat2_train,feat3_train]

feat1_test = np.loadtxt('comfeat_test0.txt')
feat2_test = np.loadtxt('comfeat_test1.txt')
feat3_test = np.loadtxt('comfeat_test2.txt')
testFeat = [feat1_test,feat2_test,feat3_test]

# change to original direcroy
os.chdir(cur_d)

tmptrainX = train[:,:-1]
trainY = train[:,-1][:,None]

tmptestX = test[:,:-1]
testY = test[:,-1][:,None]

# normalise output
meany = np.mean(trainY)
vary = np.std(trainY, ddof=1)
trainY = (trainY - meany) / vary

# check whether sid is specified
if len(sys.argv) == 3:
    sid_range = [int(sys.argv[2])]
else:
    sid_range = range(-3,4)

for sid in sid_range:
    # deal with feature alone
    if sid < 0:
        trainX = trainFeat[-sid-1]
        testX = testFeat[-sid-1]
    # deal with feature combined with raw data
    elif sid > 0:
        trainX = np.hstack((tmptrainX,trainFeat[sid-1]))
        testX = np.hstack((tmptestX,testFeat[sid-1]))
    # deal with raw data alone
    else:
        trainX = tmptrainX
        testX = tmptestX

    dim = trainX.shape[1]
    # normalise input
    tmpx = np.concatenate((trainX, testX))
    tmpx = (tmpx - np.mean(tmpx,axis=0))/np.std(tmpx,axis=0,ddof=1)
    numD = len(trainX)
    trainX = tmpx[:numD,:]
    testX = tmpx[numD:,:]

    param_file = 'param%d.txt' % sid
    params = np.loadtxt(param_file)
    # initialize the kernel with the params
    k = gp.kern.RBF(input_dim=dim, variance=params[0], lengthscale = params[1:-1], ARD=True)
    m = gp.models.GPRegression(trainX, trainY, k)
    # initialize the noise variance in the model
    m.param_array[:] = params
    m.initialize_parameter()

    # now the model is ready for prediction
    res = m.predict(testX)
    # predict mean
    print res[0] * vary + meany
    # predict variance
    print res[1] * vary * vary


