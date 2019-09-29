import numpy as np
import pandas as pd
import time
import h5py
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from threading import Lock


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


class kNN:
    def __init__(self, k, xtest, ytest, xtrain, ytrain):
        self.xtest = xtest
        self.ytest = ytest
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.k = k
        self.mutex = Lock()
        self.mutex2 = Lock()

    def NNmultiprocess(self, i):
        distlist = list()
        for j in range(self.xtrain.shape[0]):
            # 计算欧氏距离
            dist = np.linalg.norm(self.xtrain[j][:] - self.xtest[i][:])
            distlist.append((dist, self.ytrain[j]))
        distlist.sort(key=lambda x: x[0])
        # 最近邻法
        if distlist[0][1] == self.ytest[i]:
            self.mutex.acquire()
            self.count += 1
            self.mutex.release()
        # k近邻法
        classdict = {}
        for j in range(self.k):
            if distlist[j][1] in classdict.keys():
                classdict[distlist[j][1]] += 1
            else:
                classdict[distlist[j][1]] = 1
        if max(classdict.items(), key=lambda x: x[1])[0] == self.ytest[i]:
            self.mutex2.acquire()
            self.count2 += 1
            self.mutex2.release()
        else:
            p = 1
        return

    # 最近邻法，得到训练集的准确率
    def NN(self):
        self.count = 0
        self.count2 = 0
        #多线程同时计算
        pool = ThreadPool(processes=8)
        pool.map(self.NNmultiprocess, range(self.xtest.shape[0]))
        pool.close()
        return float(self.count) / float(self.xtest.shape[0]), float(self.count2) / float(self.xtest.shape[0])


iris_data = pd.read_csv('data/iris.data')
np.random.seed(int(time.time()))
sampler = np.random.permutation(iris_data.shape[0])
iris_data = iris_data.take(sampler)

irisdataarray = np.array(iris_data)
irisdatax = np.delete(irisdataarray, range(irisdataarray.shape[1]-1, irisdataarray.shape[1]), axis=1)
irisdatay = np.delete(irisdataarray, range(0, irisdataarray.shape[1]-1), axis=1)
irisxtest = irisdatax[:139][:]
irisxtrain = irisdatax[139:][:]
irisytest = irisdatay[:139][:].reshape(-1)
irisytrain = irisdatay[139:][:].reshape(-1)

knniris = kNN(3, irisxtest, irisytest, irisxtrain, irisytrain)
c1, c2 = knniris.NN()
print("Use knn, the correct rate of iris in %f,%f" % (c1, c2))

irisfisher = fisher(irisdatax, irisdatay, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
irisfishertest = irisfisher[:139][:]
irisfishertrain = irisfisher[139:][:]

knnirisfisher = kNN(3, irisfishertest, irisytest, irisfishertrain, irisytrain)
c1, c2 = knnirisfisher.NN()

print("Use knn and fisher, the correct rate of iris in %f,%f" % (c1, c2))

sonar_data = pd.read_csv('data/sonar.all-data')
sampler = np.random.permutation(sonar_data.shape[0])
sonar_data = sonar_data.take(sampler)
sonardataarray = np.array(sonar_data)
sonardatax = np.delete(sonardataarray, range(sonardataarray.shape[1]-1, sonardataarray.shape[1]), axis=1)
sonardatay = np.delete(sonardataarray, range(0, sonardataarray.shape[1]-1), axis=1)
sonarxtest = sonardatax[:186][:]
sonarxtrain = sonardatax[186:][:]
sonarytest = sonardatay[:186][:].reshape(-1)
sonarytrain = sonardatay[186:][:].reshape(-1)

knnsonar = kNN(5, sonarxtest, sonarytest, sonarxtrain, sonarytrain)
s1, s2 = knnsonar.NN()
print("Use knn, the correct rate of sonar in %f,%f" % (s1, s2))

sonarfisher = fisher(sonardatax, sonardatay, ['M', 'R'])
sonarfishertest = sonarfisher[:186][:]
sonarfishertrain = sonarfisher[186:][:]

knnsonarfisher = kNN(3, sonarfishertest, sonarytest, sonarfishertrain, sonarytrain)
s1, s2 = knnsonarfisher.NN()

print("Use knn and fisher, the correct rate of sonar in %f,%f" % (s1, s2))

with h5py.File('data/usps.h5', 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    y_te = test.get('target')[:]

#knnusps = kNN(20, X_te, y_te, X_tr, y_tr)
#c1, c2 = knnusps.NN()
