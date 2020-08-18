from sklearn.tree import DecisionTreeClassifier #导入所需模块
from sklearn.tree import export_graphviz #导入所需模块
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz

wine = load_wine()
Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)
#实例化,random_state来控制随机性，设置固定的随机模式来避免精确性的变动
clf = DecisionTreeClassifier(criterion="entropy",random_state=20,splitter="random") 
clf = clf.fit(Xtrain,Ytrain)#用训练集训练模型
score = clf.score(Xtest,Ytest)#导入测试集，从接口中调用需要的信息，给模型打分，用来判断精确性

print(score)
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜色强度','色调','稀释葡萄酒','脯氨酸']
dot_data = export_graphviz( clf
                            ,feature_names = feature_name
                            ,class_names = ["琴酒","雪莉","贝尔摩德"]
                           #颜色填充
                            ,filled=True
                           #方形框变圆形框
                            ,rounded=True
                            ,out_file=None)
graph = graphviz.Source(dot_data).view()
