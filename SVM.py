import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import RocCurveDisplay
from sklearn.decomposition import PCA

#读取数据
df = pd.read_excel(r'0614-熔融 结晶 全反射 4类-类别标签修改.xlsx')

#划分特征和类
X = df.drop(columns=['samples','classes'])
Y=df['classes']

# 将数据集分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

#定义标准化函数
def scale(x):
    # store columns in advance
    cols = x.columns
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    # avoid warning, transform it back to df
    x = pd.DataFrame(x, columns=cols)  #https://www.jianshu.com/p/fc3d3c2cd3ca
    return x

# # 归一化
# min_max_scaler = MinMaxScaler()
# X_train_normalized = min_max_scaler.fit_transform(X_train)
# X_test_normalized = min_max_scaler.fit_transform(X_test)

# 标准化
X_train_standardized = scale(X_train)
X_test_standardized = scale(X_test)

# 创建并训练模型
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_standardized, Y_train)

# 进行预测
Y_pred = model.predict(X_test_standardized)

# 打印预测结果
print(df)
# SS
# print("\n分类报告：")
# print(classification_report(Y_test, Y_pred))
# print("准确率：", accuracy_SSscore(Y_test, Y_pred))