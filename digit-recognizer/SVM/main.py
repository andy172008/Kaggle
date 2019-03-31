from numpy import *
from sklearn import svm
import csv
import datetime


# 把数组中的字符串转换成整数
def toInt(array):
    array = mat(array)
    m, n = shape(array)
    # 使用range不会生成list，性能要优于range
    for i in range(m):
        for j in range(n):
            array[i, j] = int(array[i, j])
    return array


# 把大于0的数都置为1
def nomalizing(array):
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != "0":  # 注意原csv文件中的数字也是字符串
                array[i, j] = 1
            else:
                array[i, j] = 0
    return array


def loadTrainData():
    l = []
    with open('train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)  # 42001*785  
    l.remove(l[0])  # 移除第0行，第0行是数据列名
    l = array(l)  # 将l由list型转化为numpy下的array型
    label = l[:, 0]  # label赋值为l的第0列
    data = l[:, 1:]  # data赋值为l的第1至最后一列
    return nomalizing(data), toInt(label)


def loadTestData():
    l = []
    with open('test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data = array(l)
    return nomalizing(data)


def saveResult(result, csvName):
    # 如果以wb打开，会出错
    with open(csvName, 'w') as myFile:
        myWriter = csv.writer(myFile)
        num = 1
        arr = []
        arr.append("ImageId")
        arr.append("Label")
        myWriter.writerow(arr)  # 先将列名插入第0行
        for i in result:
            tmp = []
            tmp.append(num)
            num = num + 1
            tmp.append(int(i))  ##不能是浮点数  
            myWriter.writerow(tmp)


def svcClassify(trainData, trainLabel, testData):
    svcClf = svm.SVC(
        C=5.0)  # default:C=1.0,kernel = 'rbf'. you can try kernel:‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’    
    svcClf.fit(trainData, ravel(trainLabel))
    testLabel = svcClf.predict(testData)
    saveResult(testLabel, 'sklearn_SVC_C=5.0_Result.csv')
    return testLabel


def main():
    starttime = datetime.datetime.now()
    trainData, trainLabel = loadTrainData()
    print("训练集读取完毕")
    testData = loadTestData()
    print("测试集读取完毕")
    svcClassify(trainData, trainLabel, testData)
    endtime = datetime.datetime.now()
    print("预测结束--程序总运行时间：" + str((endtime - starttime).seconds) + "秒")


main()
