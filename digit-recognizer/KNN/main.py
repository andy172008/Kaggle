from numpy import *
import operator
import csv
import os,sys


def toInt(array):
    array = mat(array)
    m, n = shape(array)
    newArray = zeros((m, n))
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])
    return newArray


def nomalizing(array):
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j] != 0:
                array[i, j] = 1
    return array


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785
    l.remove(l[0])
    l = array(l)
    label = l[:, 0]
    data = l[:, 1:]
    return nomalizing(toInt(data)), toInt(label)
    # label 1*42000  data 42000*784
    # return data,label


def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    # 28001*784
    l.remove(l[0])
    data = array(l)
    return nomalizing(toInt(data))
    # data 28000*784



# inX是所要测试的向量
# dataSet是训练样本集，一行对应一个样本。dataSet对应的标签向量为labels
# k是所选的最近邻数目

# dataSet:m*n   labels:m*1  inX:1*n
def classify(inX, dataSet, labels, k):

    inX = mat(inX)
    dataSet = mat(dataSet)

    labels = mat(labels)
    # 返回矩阵行数，也就是test样本个数
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 在array创建的矩阵，乘号代表对矩阵里的元素做操作
    sqDiffMat = array(diffMat) ** 2
    # axis为1，代表按行相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i], 0]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def saveResult(result):
    with open('result.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        order = 1
        for i in result:
            tmp = []
            tmp.append(order)
            order += 1
            tmp.append(i)
            myWriter.writerow(tmp)


def handwritingClassTest():
    print("step 0")
    print("step 0.5")
    trainData, trainLabel = loadTrainData()
    print(trainLabel.shape)
    print("step 1")
    testData = loadTestData()
    print("step 2")

    m, n = shape(testData)

    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)
        resultList.append(classifierResult)
        print("%d / %d"%(i,m))
    saveResult(resultList)

handwritingClassTest()
#saveResult([12])
#loadTrainData()
