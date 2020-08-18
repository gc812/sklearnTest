import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
#使用sklearn对数据进行标准化和归一化
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets
#找到两组数据集中最近邻点,K-近邻算法
from sklearn.neighbors import KNeighborsClassifier

#导入datingTestSet数据集
file = open("datingTestSet.txt","r")
listx = file.readlines()
dataset = []
for fields in listx: 
    fields = fields.strip();
    fields = fields.strip("[]");
    fields = fields.strip().split("\t")   
    dataset.append(fields);
data_X = [x[0:3] for x in dataset]
data_Y = [x[-1] for x in dataset]
  

data_X = preprocessing.scale(data_X)


# 把数据分开，并打乱数据顺序
X_train,X_test,Y_train,Y_test = train_test_split(data_X,data_Y,test_size = 0.3)

knn = KNeighborsClassifier()

#利用fit（）方法把需要训练的数据放进去，自动训练
knn.fit(X_train,Y_train)

print(knn.score(X_test,Y_test))
