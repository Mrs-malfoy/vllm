import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据存储在两个列表中
# rate = list(range(2, 21))  # 你的rate数据
# ttft = [69.96, 71.16, 69.16, 83.31, 75.03, 78.15, 75.52, 79, 76.14, 84.62, 
#         84.26, 93.66, 92.79, 87.3, 90.98, 86.48, 100.33, 98.66, 91.76]  # 你的ttft数据
# mean_ttft = [43.19, 42.86, 48.34, 45.26, 47.96, 49.26, 48.89, 49.34, 53.73,
            #  53.46, 52.25, 51.5, 52.56, 53.32, 53.92, 52.75, 53.28, 54.82, 52.64]
# request = 200
rate = np.arange(2, 5.5, 0.5)
mean_ttft = [42.81, 45.92, 46.78, 47.45,54.47, 57.04, 56.20]
ttft = [104.48, 114.82, 106.38, 100.31, 268.51, 373.88, 274.51]

plt.figure(figsize=(10, 6))  # 设置图的大小
plt.plot(rate, ttft, marker='o')  # 绘制线图，并添加数据点标记
plt.xlabel('Rate')  # x轴标签
plt.ylabel('TTFT')  # y轴标签
plt.title('Rate vs TTFT')  # 图表标题
plt.grid(True)  # 添加网格
# 在代码最后 plt.show() 之前添加
plt.savefig('rate_vs_ttft.png')
