import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib import font_manager
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
def load_data(filepath):
    df = pd.read_excel(filepath)
    X = df.drop(columns=['samples', 'contents'])
    Y = df['contents']
    return X, Y


# 数据预处理
def preprocess_data(X, Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
    return X_train, X_test, Y_train, Y_test


# 训练模型
def train_model(X_train, Y_train):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    model.fit(X_train, Y_train)
    return model


# 评估模型
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(Y_test, y_pred) # 平均绝对误差
    mse = mean_squared_error(Y_test, y_pred) # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    r2 = r2_score(Y_test, y_pred) # 决定系数
    return mae, mse, rmse, r2, y_pred


# 绘制预测结果图
def plot_results(Y_test, y_pred, save_path):
    plt.figure(figsize=(20, 6))
    y_test_t = np.asarray(Y_test)
    plt.plot(y_test_t[:8], label='真实值')
    plt.plot(y_pred[:8], label='预测值')
    plt.xlabel('样本索引')
    plt.ylabel('值')
    plt.legend()
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


# 绘制特征重要性图
def plot_feature_importance(model, feature_names, save_path):
    feature_importances = model.feature_importances_
    feature_importances_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importances})
    feature_importances_df = feature_importances_df.sort_values('重要性', ascending=False)

    colors = cm.viridis(np.linspace(0, 1, len(feature_names)))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importances_df['特征'], feature_importances_df['重要性'], color=colors)
    ax.invert_yaxis()

    plt.rcParams['font.sans-serif'] = ['Arial']
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    font_path = 'C:/Windows/Fonts/方正粗黑宋简体.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    ax.set_xlabel('特征重要性', fontsize=20, fontproperties=font_prop)
    ax.set_title('随机森林回归特征重要性可视化', fontsize=20, fontproperties=font_prop)

    for i, v in enumerate(feature_importances_df['重要性']):
        ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    plt.show()


# 主程序
if __name__ == "__main__":
    # 加载数据
    X, Y = load_data(r'0614-熔融 结晶 全反射-40-回归.xlsx')

    # 数据预处理
    X_train, X_test, Y_train, Y_test = preprocess_data(X, Y)

    # 训练模型
    model = train_model(X_train, Y_train)

    # 评估模型
    mae, mse, rmse, r2, y_pred = evaluate_model(model, X_test, Y_test)
    # 设置全局格式化选项，保留两位小数
    np.set_printoptions(formatter={'float': '{:.2f}'.format})

    print("Y_test:")
    print(Y_test)
    print("y_pred:")
    print(y_pred)

    # 输出评估结果
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"决定系数 (R²): {r2:.2f}")

    # 绘制预测结果图
    plot_results(Y_test, y_pred, './前8条预测结果图.jpg')

    # 绘制特征重要性图
    plot_feature_importance(model, list(X.columns), './特征重要性.jpg')