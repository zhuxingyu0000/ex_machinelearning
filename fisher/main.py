import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns


def fisher_trainingkcross(dataframe, classarray):  # 训练并且使用k折交叉验证验证训练结果，这里取k=10
    dataframe = np.array(dataframe)
    k = 10  # k折交叉验证取k=10
    num = dataframe.shape[0]  # 总数据
    classes = len(classarray)  # 类别数
    correctrate = list()  # 存放ｋ次训练的正确率
    for i in range(k):
        # 获得训练数据
        datatrain = np.delete(dataframe, range(int(i * num / k), int((i + 1) * num / k)), axis=0)
        [rows, cols] = datatrain.shape
        # 获得验证数据
        dataverify = dataframe[int(i * num / k):int((i + 1) * num / k)][:]

        # 求均值向量
        average = [np.zeros(cols - 1) for x in range(classes)]
        datatrainclassified = [[] for x in range(classes)]  # 将相同类别的向量放到相同数组里

        for j in range(rows):
            for t in range(len(classarray)):
                if datatrain[j][cols - 1] == classarray[t]:
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
                si[j] = si[j] + np.dot(x_err.reshape(-1, 1), x_err.reshape(1, -1))

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
            wx = np.dot(np.linalg.inv(sw), d.reshape(-1, 1))
            [x, y] = wx.shape
            wx = wx.reshape(y, x)

            wxs.append(wx)

            # 求阈值w0
            w0 = np.dot(wx, average[index1]) + np.dot(wx, average[index2])
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
            # 将多分类问题转换为两两分类问题
            for n in range(int(classifiers)):
                index2 = index2 + 1
                if index2 >= classes:
                    index2 = index1 + 2
                    index1 = index1 + 1
                result = np.dot(wxs[n], datainput) - w0s[n]
                if result[0] > 0:
                    correctarray[index1][0] += 1
                else:
                    correctarray[index2][0] += 1
            correctarray.sort(key=lambda x: x[0], reverse=True)
            if classarray[correctarray[0][1]] == correctclass:
                correct += 1
        correctrate.append(float(correct) / float(total))

        # 曲线绘制
        if i == 1:
            pltarray = [[[], []] for j in range(int(classifiers))]
            for j in range(dataframe.shape[0]):
                datainput = dataframe[j][:-1]
                label = dataframe[j][-1]
                index1 = 0
                index2 = 0
                for n in range(int(classifiers)):
                    index2 = index2 + 1
                    if index2 >= classes:
                        index2 = index1 + 2
                        index1 = index1 + 1
                    result = np.dot(wxs[n], datainput) - w0s[n]
                    pltarray[n][0].append(result[0])
                    pltarray[n][1].append(label)
            index1 = 0
            index2 = 0
            for n in range(int(classifiers)):
                index2 = index2 + 1
                if index2 >= classes:
                    index2 = index1 + 2
                    index1 = index1 + 1
                df = pd.DataFrame({"Class": pltarray[n][1], "ProjectionPoint": pltarray[n][0]})
                ax = sns.stripplot(x="Class", y="ProjectionPoint", data=df, jitter=True, )
                plt.title('The classification between %s and %s' % (classarray[index1], classarray[index2]))
                plt.axhline(0, color='red', linestyle='--')
                plt.show()
                x1 = df[df['Class'] == classarray[index1]]
                x2 = df[df['Class'] == classarray[index2]]
                ax2 = sns.distplot(x1["ProjectionPoint"], rug=True, hist=False, label=classarray[index1])
                ax2 = sns.distplot(x2["ProjectionPoint"], rug=True, hist=False, label=classarray[index2])
                plt.title('The classification between %s and %s' % (classarray[index1], classarray[index2]))
                plt.show()

    return np.mean(np.array(correctrate))


if (not os.path.exists('data/iris.data')) or (not os.path.exists('data/sonar.all-data')):
    print("The data doesn't exist.Please run data_fetch.sh!")
    exit(0)

iris_data = pd.read_csv('data/iris.data')
np.random.seed(int(time.time()))
sampler = np.random.permutation(iris_data.shape[0])
iris_data = iris_data.take(sampler)
correctrate = fisher_trainingkcross(iris_data, ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
print("The correct rate in iris data is:", correctrate)

sonar_data = pd.read_csv('data/sonar.all-data')
sampler = np.random.permutation(sonar_data.shape[0])
sonar_data = sonar_data.take(sampler)
correctrate = fisher_trainingkcross(sonar_data, ['M', 'R'])

print("The correct rate in sonar data is:", correctrate)
