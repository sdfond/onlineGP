function [] = rawTrain(i)
load('sonarData.mat');

tx = sigTrain(:,1:end-1);
ex = sigTest(:,1:end-1);
ty = sigTrain(:,end);
meany = mean(ty);
vary = sqrt(var(ty));
ty = ty - meany;
ty = ty / vary;
ey = sigTest(:,end);

if i > 0
    len = size(trainX{i},2);
    tx(:,end+1:end+len) = trainX{i};
    ex(:,end+1:end+len) = testX{i};
elseif i < 0
    tx = trainX{-i};
    ex = testX{-i};
end

loghyp.cov = zeros(1,size(tx, 2)+1);
for j = 1:size(tx,2)
    avg1 = log(2) - 2*log(sqrt(mean(abs(diff(tx(:,j))))));
    avg2 = log(2) - 2*log(0.1*sqrt(mean(abs(diff(tx(:,j))))));
    if mod(j,2) == 1
        loghyp.cov(j) = max(avg1, avg2) + rand;
    else
        loghyp.cov(j) = max(avg1, avg2) + rand;
    end
end
loghyp.cov(end) = log(1);
loghyp.lik = log(2) - 2*log(sqrt(mean(abs(diff(ty)))));
loghyp.mean = [];


% start training the hyper-paramters
% third parameter is -1000 where 1000 indicates the maxinum iteration time
loghyp = minimize(loghyp, @gp, -500, @infExact, @meanZero, @covSEard, @likGauss, tx, ty);
% doing prediction on the test data, return the predict mean and variance
[m s2] = gp(loghyp, @infExact, @meanZero, @covSEard, @likGauss, tx, ty, ex);
% since training data is normalised, in order to get the real mean and
% real variance, need to scale them back
m = m * vary + meany;
s2 = s2 * vary * vary;
fname = sprintf('res%d.mat', i);


tmpm = m > meany;
tmpy = ey > meany;
mislabel = [length(find(tmpm(find(tmpy==0)) ~= tmpy(find(tmpy==0)))) ...
    length(find(tmpm(find(tmpy==1)) ~= tmpy(find(tmpy==1))))]
save(fname, 'loghyp', 'm', 's2', 'mislabel');