import re
import matplotlib.pyplot as plt
import numpy as np

# 用于存储解析出的数据
prefills = []
decodes = []
total_compute = []
gpu_usage = []
cpu_usage = []
exec_time = []

# 解析日志文件
with open('output.log', 'r') as f:
    for line in f:
        if 'Schedule stats' in line:
            prefill_match = re.search(r'Prefills: (\d+)', line)
            decode_match = re.search(r'Decodes: (\d+)', line)
            gpu_match = re.search(r'GPU cache usage: ([\d.]+)', line)
            cpu_match = re.search(r'CPU cache usage: ([\d.]+)', line)
            time_match = re.search(r'Execution time: ([\d.]+)', line)
            
            if all([prefill_match, decode_match, gpu_match, cpu_match, time_match]):
                prefills.append(int(prefill_match.group(1)))
                decodes.append(int(decode_match.group(1)))
                total_compute.append(int(prefill_match.group(1)) + int(decode_match.group(1)))
                gpu_usage.append(float(gpu_match.group(1)))
                cpu_usage.append(float(cpu_match.group(1)))
                exec_time.append(float(time_match.group(1)))

# 创建x轴数据
x = range(len(prefills))

# 设置图表样式
# plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300

# 1. Prefills图
plt.figure()
plt.plot(x, prefills, 'b-', linewidth=2)
plt.title('Number of Prefills over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Number of Prefills', fontsize=12)
plt.grid(True)
plt.savefig('prefills.png', bbox_inches='tight')
plt.close()

# 2. Decodes图
plt.figure()
plt.plot(x, decodes, 'r-', linewidth=2)
plt.title('Number of Decodes over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Number of Decodes', fontsize=12)
plt.grid(True)
plt.savefig('decodes.png', bbox_inches='tight')
plt.close()

# 3. Total Compute图
plt.figure()
plt.plot(x, total_compute, 'g-', linewidth=2)
plt.title('Total Compute over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Total Compute', fontsize=12)
plt.grid(True)
plt.savefig('total_compute.png', bbox_inches='tight')
plt.close()

# 3. GPU Usage图
plt.figure()
plt.plot(x, gpu_usage, 'g-', linewidth=2)
plt.title('GPU Cache Usage over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('GPU Cache Usage (%)', fontsize=12)
plt.grid(True)
plt.savefig('gpu_usage.png', bbox_inches='tight')
plt.close()

# 4. CPU Usage图
plt.figure()
plt.plot(x, cpu_usage, 'm-', linewidth=2)
plt.title('CPU Cache Usage over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('CPU Cache Usage (%)', fontsize=12)
plt.grid(True)
plt.savefig('cpu_usage.png', bbox_inches='tight')
plt.close()

# 5. Execution Time图
plt.figure()
plt.plot(x, exec_time, 'y-', linewidth=2)
plt.title('Execution Time over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.grid(True)
plt.savefig('exec_time.png', bbox_inches='tight')
plt.close()

# 打印统计信息
print(f"总调用次数: {len(prefills)}")
print(f"Prefills - 平均: {np.mean(prefills):.2f}, 最大: {max(prefills)}, 最小: {min(prefills)}")
print(f"Decodes - 平均: {np.mean(decodes):.2f}, 最大: {max(decodes)}, 最小: {min(decodes)}")
print(f"GPU使用率 - 平均: {np.mean(gpu_usage):.2f}%, 最大: {max(gpu_usage):.2f}%, 最小: {min(gpu_usage):.2f}%")
print(f"CPU使用率 - 平均: {np.mean(cpu_usage):.2f}%, 最大: {max(cpu_usage):.2f}%, 最小: {min(cpu_usage):.2f}%")
print(f"执行时间 - 平均: {np.mean(exec_time):.2f}ms, 最大: {max(exec_time):.2f}ms, 最小: {min(exec_time):.2f}ms")