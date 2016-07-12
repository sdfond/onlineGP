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
if len(sys.argv) == 4:
    sid_range = [int(sys.argv[3])]
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

    # write the training and testing data
    gpTrain = np.hstack((trainX, trainY))
    gpTest = np.hstack((testX, testY))
    savename = 'gpTrain%d.txt' % sid
    np.savetxt(savename, gpTrain)
    savename = 'gpTest%d.txt' % sid
    np.savetxt(savename, gpTest)


    # setting up a initial guess of hyper-parameters
    avg1 = [2 / (np.sqrt(np.mean(np.fabs(np.diff(trainX[:,i])))))**2 + np.random.normal(0,5) for i in range(trainX.shape[1])]
    avg2 = [2 / (0.1*np.sqrt(np.mean(np.fabs(np.diff(trainX[:,i])))))**2 + np.random.normal(0,5) for i in range(trainX.shape[1])]
    avg = np.maximum(avg1,avg2)

    # define a kernel function (RBF ard kernel)
    # 4 parameters: input data dimension, initial value of signal variance, initial value of length-scales, whether it's RBF-ARD kernel
    k = gp.kern.RBF(dim,10.0,avg,True)

    # get a GPRegression model m, set the restart runs as 3
    m = gp.models.GPRegression(trainX, trainY, k)
    print 'optimization method: %s' % sys.argv[2]
    if sys.argv[2] == 'bfgs':
        m.optimize('bfgs')
    else:
        m.optimize_restarts(3)


    # obtain the prediction result on testX, res contains 4 arrays: predict mean, predict variance, lower and up 95% confident interval
    res = m.predict(testX)
    # res[0] contains predict mean
    ym = res[0] * vary + meany
    # res[1] contains predict variance
    ys = res[1] * vary * vary

    resname = "res%d.txt" % sid
    resf = open(resname, 'w+')

    # compare with the real value, display the results
    meantest = meany
    s1 = np.where(testY > meantest)[0]
    print "number of >%dm samples: %d" % (meantest, s1.size)
    s2 = np.where(ym[s1] > meantest)[0]
    print "number of >%dm correctly labeled: %d" % (meantest, s2.size)
    resf.write("%d %d\n" % (len(s1), len(s2)))
    s1 = np.where(testY < meantest)[0]
    print "number of <%dm samples: %d" % (meantest, s1.size)
    s2 = np.where(ym[s1] < meantest)[0]
    print "number of <%dm correctly labeled: %d" % (meantest, s2.size)
    resf.write("%d %d\n" % (len(s1), len(s2)))
    for i in range(0,len(ym)):
        resf.write("%lf %lf\n" % (ym[i], ys[i]))
    resf.close()

    # use kernel k to generate any covariance matrix
    k = m.kern
    C = k.K(testX,trainX)

    res = m.param_array.tolist()
    print m

    hypname = "res-hyp%d.txt" % sid
    hypf = open(hypname, 'w+')

    signal = res[0]
    signal = 0.5 * np.log(signal)
    noise = res[-1]
    noise = 0.5 * np.log(noise)

    hypf.write("%lf %lf %lf %lf %d %d %d\n" % (signal, noise, meany, vary, trainX.shape[1], trainX.shape[0], testX.shape[0]))
    res = res[1:-1]
    for item in res:
        hypf.write("%lf\n" % np.log(item))
    hypf.close()

