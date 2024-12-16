import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib
from openpyxl import load_workbook
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import font_manager

#读取数据
df = pd.read_excel(r'0614-熔融 结晶 全反射-回归.xlsx')
#划分特征和类
X = df.drop(columns=['samples','contents'])
Y=df['contents']

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 将数据集分为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
# 初始化随机森林回归模型
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
# 训练模型
model.fit(X_train, Y_train)
# 使用测试集进行预测
y_pred = model.predict(X_test)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# 计算评估指标
mae = "{:.2f}".format(mean_absolute_error(Y_test, y_pred))
mse = "{:.2f}".format(mean_squared_error(Y_test, y_pred))
r2 = "{:.2f}".format(r2_score(Y_test, y_pred))

print(f"平均绝对误差: {mae}")
print(f"均方误差: {mse}")
print(f"决定系数（R\u00b2）: {r2}")
print("Y_test: \n", Y_test)
print("y_pred: \n", y_pred.reshape(-1, 1))

# 绘制线图
plt.figure(figsize=(20, 6))
y_test_t = np.asarray(Y_test)
plt.plot(y_test_t[:8], label='True Values')
plt.plot(y_pred[:8], label='Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.savefig('./前8条预测结果图.jpg', dpi=400, bbox_inches='tight')#保存图形
plt.show()
'''
版权声明：本文为博主原创文章，遵循
CC
4.0
BY - SA
版权协议，转载请附上原文出处链接和本声明。

原文链接：https: // blog.csdn.net / weixin_51094405 / article / details / 130512307
# 提取特征重要性
feature_importances = model.feature_importances_
'''
# 获取特征重要性得分
feature_importances = model.feature_importances_
# 创建特征名列表
feature_names = list(X.columns)
# 创建一个DataFrame，包含特征名和其重要性得分
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
# 对特征重要性得分进行排序
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
# 颜色映射
colors = cm.viridis(np.linspace(0, 1, len(feature_names)))

# 可视化特征重要性
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
# 设置支持中文的字体，如 Arial
plt.rcParams['font.sans-serif'] = ['Arial']  #如果不设置字体，会在Python环境找不到字体文件
# 设置 x 轴刻度标签的字体大小
ax.tick_params(axis='x', labelsize=12)  # 设置 x 轴刻度标签字体大小为 12
# 设置 y 轴刻度标签的字体大小
ax.tick_params(axis='y', labelsize=12)  # 设置 y 轴刻度标签字体大小为 12
# 设置轴标签和标题字体
font_path = 'C:/Windows/Fonts/方正粗黑宋简体.ttf'  # Windows 系统中的字体路径
font_prop = font_manager.FontProperties(fname=font_path)
ax.set_xlabel('特征重要性', fontsize=20, fontproperties=font_prop)  # 图形x轴标签的字体大小和类型
ax.set_title('随机森林回归特征重要性可视化', fontsize=20, fontproperties=font_prop)  #图形标题的字体大小和类型
for i, v in enumerate(feature_importances_df['importance']):
    ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

# # 设置图形样式
# plt.style.use('default')
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
# ax.spines['left'].set_linewidth(0.5)#左边框粗细
# ax.spines['bottom'].set_linewidth(0.5)#下边框粗细
# ax.tick_params(width=0.5)#刻度宽度
# ax.set_facecolor('white')#背景色为白色
# ax.grid(False)#关闭内部网格线

# 保存图形
plt.savefig('./特征重要性.jpg', dpi=400, bbox_inches='tight')
plt.show()

# # 绘制特征重要性（基础版）
# plt.barh(range(len(feature_importances)), feature_importances)
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature Index')
# plt.title('Feature Importance in Random Forest Regressor')
# plt.show()

