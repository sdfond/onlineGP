function [] = createData(dName)

cfolder = pwd;
cd(dName);
sigTrain = load('rawTrain.txt');
sigTest = load('rawTest.txt');
trainX = cell(1,3);
testX = cell(1,3);
for i = 0:2
    trainName = sprintf('comfeat_train%d.txt', i);
    testName = sprintf('comfeat_test%d.txt', i);
    trainX{i+1} = load(trainName);
    testX{i+1} = load(testName);
end
cd(cfolder);
save('sonarData.mat', 'trainX', 'testX', 'sigTrain', 'sigTest');
