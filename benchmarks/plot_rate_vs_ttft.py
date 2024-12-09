import matplotlib.pyplot as plt
import numpy as np

# 假设你的数据存储在两个列表中
# rate = list(range(2, 21))  # 你的rate数据
# ttft = [69.96, 71.16, 69.16, 83.31, 75.03, 78.15, 75.52, 79, 76.14, 84.62, 
#         84.26, 93.66, 92.79, 87.3, 90.98, 86.48, 100.33, 98.66, 91.76]  # 你的ttft数据
# mean_ttft = [43.19, 42.86, 48.34, 45.26, 47.96, 49.26, 48.89, 49.34, 53.73,
            #  53.46, 52.25, 51.5, 52.56, 53.32, 53.92, 52.75, 53.28, 54.82, 52.64]
# request = 200
rate = np.arange(5, 10.5, 0.5)
# b_mean_ttft = [42.81, 45.92, 46.78, 47.45,54.47, 57.04, 56.20]
# b_ttft = [104.48, 114.82, 106.38, 100.31, 268.51, 373.88, 274.51]
benchmark_mean_ttft = [61.60, 71.34, 65.94, 72.52, 67.35, 95.46, 430.13, 577.67, 953.78, 1551.37, 2015.67]
benchmark_ttft = [270.24, 549.01, 322.81, 311.26, 340.47, 844.11, 4791.76, 5005.14, 5861.89, 7808.40, 9825.04]
# our algorithm
# mean_ttft = [40.38, 42.49, 44.29, 44.40, 46.03, 48.38, 49.99]
# ttft = [81.49, 81.71, 86.62, 78.26, 87.17, 72.53, 79.97]
new_mean_ttft = [59.71, 58.61, 59.64, 68.51, 67.44, 80.92, 172.19, 264.40, 329.03, 340.59, 386.70]
new_ttft = [299.94, 141.73, 132.91, 317.99, 225.62, 473.71, 1104.74, 1122.70, 1247.07, 1152.13, 1333.44]

plt.figure(figsize=(10, 6))  # 设置图的大小
plt.plot(rate, benchmark_mean_ttft, marker='o', label='Baseline', color='red')  # 绘制线图，并添加数据点标记
plt.plot(rate, new_mean_ttft, marker='s', label='Our Method', color='blue')  # 绘制线图，并添加数据点标记
plt.xlabel('Rate')  # x轴标签
plt.ylabel('Mean TTFT')  # y轴标签
plt.title('Rate vs meanTTFT')  # 图表标题
plt.grid(True)  # 添加网格
plt.legend()  # 确保调用了这行代码
plt.savefig('benchmarks/rate_vs_mean_ttft.png')

