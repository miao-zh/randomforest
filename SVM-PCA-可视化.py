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
from matplotlib.colors import ListedColormap


#读取数据
df = pd.read_excel(r'0614-熔融 结晶 全反射 3类-类别标签修改.xlsx')

#划分特征和类
X = df.drop(columns=['samples','classes'])
Y=df['classes']

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
X_standardized = scale(X)

#pca降维以方便可视化
pca = PCA(n_components=2)
X_pca=pca.fit_transform(X) # 降维后的结果

# 分类标签列表
print('Class labels:', np.unique(Y))

# 将数据集分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=42, stratify=Y)

print('Labels counts in Y:', np.bincount(Y))  # 原数据集中各分类标签出现次数 [24 24 24]
print('Labels counts in Y_train:', np.bincount(Y_train))  # 训练集中各分类标签出现次数 [35 35 35]
print('Labels counts in Y_test:', np.bincount(Y_test))   # 测试集中各分类标签出现次数 [ ]



# 绘制决策边界图 函数
def plot_decision_regions(X, Y, classifier, test_idx=None, resolution=0.02):
    # 设置标记生成器和颜色图
    markers = ('s', '^', 'o', 'x', 'v')                     # 标记生成器
    colors = ('red', 'blue', 'light green', 'gray', 'cyan')  # 定义颜色图
    cmap = ListedColormap(colors[:len(np.unique(Y))])  #自定义颜色图

# 创建并训练模型
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, Y_train)

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap='coolwarm', alpha=0.8)
# 绘制训练数据点
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k', marker='o', s=100)
plt.scatter(X_test[:, 0], X_test[:, 1], c='black', edgecolors='k', marker='x', s=100, label='Test data')
plt.show()

# 进行预测
Y_pred = model.predict(X_test)

# 打印预测结果及模型评分
print("Predicted labels: ", Y_pred)
print('Misclassified samples: %d' % (Y_test != Y_pred).sum())   # 输出错误分类的样本数
#利用classifier.score（）分别计算训练集和测试集的准确率。
train_score = model.score(X_train,Y_train)
print("训练集准确率：",train_score)
test_score = model.score(X_test,Y_test)
print("测试集准确率：",test_score)

# 评估模型
print("混淆矩阵：")
print(confusion_matrix(Y_test, Y_pred))
#
print("\n分类报告：")
print(classification_report(Y_test, Y_pred))

print("准确率：", accuracy_score(Y_test, Y_pred))