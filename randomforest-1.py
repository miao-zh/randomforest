import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib
from openpyxl import load_workbook

#读取数据
df = pd.read_excel(r'0614-熔融 结晶 全反射 3类.xlsx')


# df = df.drop(df.columns[0,1],axis=1)
# df.to_excel(r'熔融 结晶 全反射 4类-1.xlsx')
# X = df.drop(columns=['samples','classes'])
# Y = df['classes']
# print(X.head())
# print(Y.head())

#划分特征和类
X = df.drop(columns=['samples','classes'])
Y=df['classes']

# 将数据集分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# 3. 构建随机森林分类器模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
# 训练模型
clf.fit(X_train, Y_train)

# 4. 使用测试集进行预测
Y_pred = clf.predict(X_test)

# 5. 评估模型准确率
accuracy = accuracy_score(Y_test, Y_pred)
print(f"模型准确率: {accuracy:.2f}")
