import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.drop(columns=['samples', 'contents'])  # 特征
    Y = df['contents']  # 目标变量
    return X, Y


# 数据预处理
def preprocess_data(X, Y):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test


# 训练支持向量机回归模型
def train_svr(X_train, Y_train):
    # 初始化 SVR 模型，使用径向基函数（RBF）作为核函数
    model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

    # 训练模型
    model.fit(X_train, Y_train)
    return model


# 评估模型
def evaluate_model(model, X_test, Y_test):
    # 使用模型进行预测
    y_pred = model.predict(X_test)

    # 计算评估指标
    mae = mean_absolute_error(Y_test, y_pred)  # 平均绝对误差
    mse = mean_squared_error(Y_test, y_pred)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    r2 = r2_score(Y_test, y_pred)  # 决定系数

    return mae, mse, rmse, r2, y_pred


# 绘制预测结果图
def plot_results(Y_test, y_pred, save_path):
    plt.figure(figsize=(20, 6))

    # 绘制真实值曲线
    plt.plot(Y_test.values[:8], label='真实值', marker='o')

    # 绘制预测值曲线
    plt.plot(y_pred[:8], label='预测值', marker='x')

    # 设置图形标题和标签
    plt.title('支持向量机回归预测结果', fontsize=16)
    plt.xlabel('样本索引', fontsize=14)
    plt.ylabel('值', fontsize=14)
    plt.legend()

    # 保存图形
    plt.savefig(save_path, dpi=400, bbox_inches='tight')

    # 显示图形
    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载数据
    X, Y = load_data(r'0614-熔融 结晶 全反射-40-回归.xlsx')

    # 数据预处理
    X_train, X_test, Y_train, Y_test = preprocess_data(X, Y)

    # 训练支持向量机回归模型
    model = train_svr(X_train, Y_train)

    # 评估模型
    mae, mse, rmse, r2, y_pred = evaluate_model(model, X_test, Y_test)
    # 设置全局格式化选项，保留两位小数
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    # 输出评估结果
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"决定系数 (R²): {r2:.2f}")

    # 绘制预测结果图
    plot_results(Y_test, y_pred, './svr_prediction_results.jpg')