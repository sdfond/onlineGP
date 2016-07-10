import re
import csv
import sys

fileName = ['fvA', 'fvB', 'fvC']
    
for i in range(0,3):
    trainName = "train-NUS.%s.txt" % fileName[i]
    trainFile = open(trainName, 'r')

    fname = "processed.data.%s-NUS_%s.csv" % (sys.argv[1], fileName[i])
    rawTrain = open('rawTrain.txt', 'wa')
    rawTest = open('rawTest.txt', 'wa')
    rawFile = csv.reader(open(fname, 'rb'), delimiter = "\t")
    trainFile = [a for a in trainFile]
    raw = [a for a in rawFile]
    raw.pop(0)

    spec = trainFile[-1]
    ist = spec.find("test_json_file")
    ien = spec.find("json-settings")
    tmps = spec[ist:ien]

    pat = re.compile("\[(.+)\]")
    m = pat.search(tmps)
    tname = [a for a in m.group(1).split(',')]
    trainFile.pop()


    ftrainName = "comfeat_train%d.txt" % (i)
    ftestName = "comfeat_test%d.txt" % (i)

    comfeatTrain = open(ftrainName, 'w+')
    comfeatTest = open(ftestName, 'w+')

    for item in trainFile:
        # extract the combined feature
        rawL = item[14:]
        featL = item[30:]
        outer = re.compile("\[(.+)\]")
        m = outer.search(featL)
        inner_str = m.group(1)
        res = re.sub('[\[,\]]', '', inner_str)
        # extract the joint ID
        tmp = rawL.split(']')[0]
        tmpnum = [int(a) for a in tmp.split(',')]

        total = sum(tname[i].rstrip('\n') in item for i in range(len(tname)))
        height = 0;

        for j in range(0,3):
            rid = tmpnum[j] - 1
            tmp1 = raw[rid][1].split(',')
            tmp2 = raw[rid][2].split(',')
            height = raw[rid][5]

            for k in range(len(tmp1)):
                if total > 0:
                    rawTest.write('%s %s ' % (tmp1[k], tmp2[k]))
                else:
                    rawTrain.write('%s %s ' % (tmp1[k], tmp2[k]))

        if total > 0:
            rawTest.write(height + '\n')
        else:
            rawTrain.write(height + '\n')

        if total > 0:
            comfeatTest.write(res + '\n')
        else:
            comfeatTrain.write(res + '\n')

    comfeatTrain.close()
    comfeatTest.close()


