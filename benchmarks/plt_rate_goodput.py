import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# 数据
x = np.array([4, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 12])
y1 = np.array([3.65, 5.36, 5.79, 6.18, 6.57, 6.92, 7.27, 7.53, 8.00, 6.01, 5.18])
y2 = np.array([3.57, 5.21, 5.63, 6.00, 6.35, 6.67, 6.98, 7.22, 5.48, 2.25, 1.11])

# 创建更密集的点以实现平滑效果
# X_smooth = np.linspace(x.min(), x.max(), 300)
# spl1 = make_interp_spline(x, y1, k=3)  # k=3 表示三次样条
# spl2 = make_interp_spline(x, y2, k=3)
# y1_smooth = spl1(X_smooth)
# y2_smooth = spl2(X_smooth)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制平滑曲线
plt.plot(x, y1, label='our_method', color='blue', linewidth=2)
plt.plot(x, y2, label='baseline', color='red', linewidth=2)

# 添加原始数据点
plt.scatter(x, y1, color='blue', s=50)
plt.scatter(x, y2, color='red', s=50)

# 设置图表属性
plt.xlabel('Issue Rate (QPS)', fontsize=12)
plt.ylabel('Goodput (QPS)', fontsize=12)
plt.title('Issue Rate vs Goodput', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 设置坐标轴范围
plt.xlim(3, 13)
plt.ylim(0, 8)

# 显示图形
plt.tight_layout()
plt.savefig('./rate_vs_goodput.png')
print("success")
