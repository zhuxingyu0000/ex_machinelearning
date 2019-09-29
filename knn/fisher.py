import numpy as np
import pandas as pd
import time


def fisher(dataxtotal, dataytotal, classarray):
    classes = len(classarray)
    [rows, cols] = dataxtotal.shape
    average = [np.zeros(cols) for x in range(len(classarray))]
    dataclassified = [[] for x in range(len(classarray))]  # 将相同类别的向量放到相同数组里

    # 求均值向量
    for j in range(rows):
        for t in range(classes):
            if dataytotal[j] == classarray[t]:
                average[t] = average[t] + dataxtotal[j]
                dataclassified[t].append(dataxtotal[j])
                break

    for j in range(classes):
        average[j] /= rows

    # 求每个类的类内离散度矩阵si
    si = list()
    for j in range(classes):
        si.append(np.zeros((cols, cols)))
        for t in range(len(dataxtotal[j])):
            x_err = dataxtotal[j][t] - average[j]
            si[j] = si[j] + np.dot(x_err.reshape(-1, 1), x_err.reshape(1, -1))

    classifiers = classes * (classes - 1) / 2  # 多类拆分为两类
    index1 = 0
    index2 = 0

    dataxnews = list()

    for n in range(int(classifiers)):
        index2 = index2 + 1
        if index2 >= classes:
            index2 = index1 + 2
            index1 = index1 + 1

        # 求总样本类内离散度矩阵sw
        sw = si[index1] + si[index2]
        sw = np.array(sw, dtype="float")

        # 求最佳变换向量wx
        d = average[index1] - average[index2]
        wx = np.dot(np.linalg.pinv(sw), d.reshape(-1, 1))

        # 投影
        dataxnew = np.dot(dataxtotal, wx)
        dataxnews.append(dataxnew)

    return np.concatenate(tuple(dataxnews), axis=1)


iris_data = pd.read_csv('data/iris.data')
np.random.seed(int(time.time()))
sampler = np.random.permutation(iris_data.shape[0])
iris_data = iris_data.take(sampler)
dataarray = np.array(iris_data)
datax = np.delete(dataarray, range(dataarray.shape[1]-1, dataarray.shape[1]), axis=1)
datay = np.delete(dataarray, range(0, dataarray.shape[1]-1), axis=1)
m = fisher(datax, datay, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
