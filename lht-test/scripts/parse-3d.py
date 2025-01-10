import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从CSV文件读取数据
data = pd.read_csv('output.csv', header=None)

# data[1] = data[1] * 64 * 128

# 提取x, y, z
x = data[0]
y = data[1]
# print(y)
z = data[2]

# 计算平均值
unique_xy = data[data[1] != 0][[0, 1]].drop_duplicates()
mean_z = []

# print(unique_xy)
for _, xy in unique_xy.iterrows():
    # 找到对应的z值并计算平均
    z_values = z[(x == xy[0]) & (y == xy[1])]
    mean_z.append(np.mean(z_values))

mean_z = np.array(mean_z)

# 提取平均值对应的x, y
x_avg = unique_xy[0].to_numpy()
y_avg = unique_xy[1].to_numpy()

# 进行线性拟合
A = np.vstack([x_avg, y_avg]).T
coeffs, residuals, rank, s = np.linalg.lstsq(A, mean_z, rcond=None)
a, b = coeffs

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(x_avg, y_avg, mean_z, c='r', marker='o')



# 绘制拟合平面
x_range = np.linspace(256, 2048, 10)
y_range = np.linspace(min(y_avg), max(y_avg), 10)
X, Y = np.meshgrid(x_range, y_range)
Z = a * X + b * Y

# 绘制每个点的垂直线
for i in range(len(x_avg)):
    ax.plot(
        [x_avg[i], x_avg[i]], 
        [y_avg[i], y_avg[i]], 
        [a*x_avg[i]+b*y_avg[i], mean_z[i]], 
        c='b', 
        linestyle='--', 
        linewidth=1,  # 线宽
        alpha=0.5     # 透明度
    )

# 分析a*x_avg[i]+b*y_avg[i], mean_z[i]的平均误差
error = np.abs(a*x_avg+b*y_avg-mean_z)
print("mean error: ", np.mean(error))

ax.plot_surface(X, Y, Z, alpha=0.5, color='g', rstride=100, cstride=100)

# 设置标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# ax.set_xlim(256, 2048)

# 反转Y轴和X轴
ax.invert_yaxis()
ax.invert_xaxis()

# 设置X轴范围


# 设置视角
ax.view_init(elev=20, azim=30)  # 调整视角以更好地查看

# 显示图形
# plt.show()
plt.savefig("perf-3d.png")

# 输出拟合系数
print(f"拟合方程: z = {a:.8f} * bs + {b:.8f} * token_count")
