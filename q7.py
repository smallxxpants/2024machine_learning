import numpy as np
import matplotlib.pyplot as plt
import os

# 数据路径
data_dir = r'C:\Users\deng2\Desktop\mlhw\hw2\HW2_programing_exercise\HW2_programing_exercise\Exercise-5-data'

# 从文件中读取数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

# 基函数
def gaussian_basis(x, mu):
    return np.exp(-(x - mu) ** 2)

# 生成基函数矩阵 Φ
def phi(x, mu_list):
    return np.column_stack([gaussian_basis(x, mu) for mu in mu_list])

# 计算 w 的闭式解
def compute_w(phi, y, lambd):
    return np.linalg.inv(phi.T @ phi + lambd * np.eye(phi.shape[1])) @ phi.T @ y

# 绘制函数
def plot_fits(x_train, y_train_list, lambd_values, mu_list):
    x_test = np.linspace(-1, 1, 100)  # 用于预测的测试数据点
    for lambd in lambd_values:
        plt.figure(figsize=(10, 6))
        for l, (x_train, y_train) in enumerate(y_train_list[:25]):  # 只绘制前25个拟合
            phi_train = phi(x_train, mu_list)
            w = compute_w(phi_train, y_train, lambd)
            phi_test = phi(x_test, mu_list)
            y_pred = phi_test @ w
            plt.plot(x_test, y_pred, label=f"Fit {l+1}", alpha=0.5)
        plt.scatter(x_train, y_train, color='red', label="Training Points", zorder=5)
        plt.title(f"Fits for log10(λ) = {np.log10(lambd)}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

# 设置mu的值
mu_list = 0.2 * (np.arange(1, 25) - 12.5)

# 加载训练数据 (假设数据按照 λ 分类存储，用户需要调整读取文件名的规则)
y_train_list = []
#for i in range(1, 101):  # 假设有100个数据集
file_path = os.path.join(data_dir, f"data_{1}")
x_train, y_train = load_data(file_path)
y_train_list.append((x_train, y_train))

# λ值 (log10(λ) 的值为 -10, -5, -1, 1)
lambd_values = [10**(-10), 10**(-5), 10**(-1), 10**(1)]

# 绘制拟合函数
plot_fits(x_train, y_train_list, lambd_values, mu_list)
