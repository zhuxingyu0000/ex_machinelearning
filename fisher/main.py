import numpy as np
import pandas as pd
import os
import time

class Fisher_classifier:
    def __init__(self, dataframeinit,classarrayinit):
        self.dataframe = np.array(dataframeinit)
        self.classarray = classarrayinit

    def trainingkcross(self):  # 训练并且使用k折交叉验证验证训练结果
        k = 10  # k折交叉验证取k=10
        num = self.dataframe.shape[0]  # 总数据
        classes = len(self.classarray)
        correctrate = list()  # 存放ｋ次训练的正确率
        for i in range(k):
            # 获得训练数据
            datatrain = np.delete(self.dataframe, range(int(i * num / k), int((i + 1) * num / k)), axis=0)
            [rows, cols] = datatrain.shape
            # 获得验证数据
            dataverify = self.dataframe[int(i * num / k):int((i + 1) * num / k)][:]

            # 求均值向量
            average = [np.zeros(cols - 1) for x in range(classes)]
            datatrainclassified = [[] for x in range(classes)]  # 将相同类别的向量放到相同数组里
            for j in range(rows):
                for t in range(len(self.classarray)):
                    if datatrain[j][cols - 1] == self.classarray[t]:
                        average[t] = average[t] + datatrain[j][:-1]
                        datatrainclassified[t].append(datatrain[j][:-1])
                        break

            for j in range(classes):
                average[j] /= rows

            # 求每个类的类内离散度矩阵si
            si = list()
            for j in range(classes):
                si.append(np.zeros((cols - 1, cols - 1)))
                for t in range(len(datatrainclassified[j])):
                    x_err = datatrainclassified[j][t] - average[j]
                    si[j] = si[j] + np.matmul(x_err.reshape(-1, 1), x_err.reshape(1, -1))

            classifiers = classes * (classes - 1) / 2  # 多类拆分为两类
            index1 = 0
            index2 = 0

            wxs = list()
            w0s = list()

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
                wx = np.matmul(np.linalg.inv(sw), d.reshape(-1, 1))
                [x, y] = wx.shape
                wx = wx.reshape(y, x)

                wxs.append(wx)

                # 求阈值w0
                w0 = np.matmul(wx, average[index1]) + np.matmul(wx, average[index2])
                w0 = w0 / 2

                w0s.append(w0)

            # 开始验证训练结果
            total = dataverify.shape[0]
            correct = 0
            for j in range(total):
                datainput = dataverify[j][:-1]
                correctclass = dataverify[j][-1]
                correctarray = [[0, i] for i in range(classes)]
                index1 = 0
                index2 = 0
                for n in range(int(classifiers)):
                    index2 = index2 + 1
                    if index2 >= classes:
                        index2 = index1 + 2
                        index1 = index1 + 1
                    result = np.matmul(wxs[n], datainput) - w0s[n]
                    if result[0] > 0:
                        correctarray[index1][0] += 1
                    else:
                        correctarray[index2][0] += 1
                correctarray.sort(key=lambda x: x[0], reverse=True)
                if self.classarray[correctarray[0][1]] == correctclass:
                    correct += 1
            correctrate.append(float(correct) / float(total))
        return np.mean(np.array(correctrate))


if (not os.path.exists('data/iris.data')) or (not os.path.exists('data/sonar.all-data')):
    print("The data doesn't exist.Please run data_fetch.sh!")
    exit(0)

iris_data = pd.read_csv('data/iris.data')
np.random.seed(int(time.time()))
sampler = np.random.permutation(iris_data.shape[0])
iris_data = iris_data.take(sampler)
f = Fisher_classifier(iris_data, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print("The correct rate in iris data is:", f.trainingkcross())

sonar_data = pd.read_csv('data/sonar.all-data')
sampler = np.random.permutation(sonar_data.shape[0])
sonar_data = sonar_data.take(sampler)
s = Fisher_classifier(sonar_data, ['M', 'R'])

print("The correct rate in sonar data is:", s.trainingkcross())
