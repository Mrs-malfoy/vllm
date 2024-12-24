import re
import matplotlib.pyplot as plt
import numpy as np
import sys

script_name = sys.argv[0]
log_path = sys.argv[1]
# 用于存储解析出的数据
prefills = []
decodes = []
total_compute = []
gpu_usage = []
cpu_usage = []
schedule_time = []
exec_time = []

# 解析日志文件
with open(log_path+"/vllm.log", 'r') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        if 'Schedule stats' in lines[i]:
            # 从当前行解析统计信息
            prefill_match = re.search(r'Prefills: (\d+)', lines[i])
            # print(prefill_match)
            # exit()
            decode_match = re.search(r'Decodes: (\d+)', lines[i])
            gpu_match = re.search(r'GPU cache usage: ([\d.]+)', lines[i])
            cpu_match = re.search(r'CPU cache usage: ([\d.]+)', lines[i])
            
            # 从下一行解析时间信息
            if i + 1 < len(lines):
                schedule_time_match = re.search(r'Schedule iteration took ([\d.]+)', lines[i+1])
                exec_time_match = re.search(r'Execute model took ([\d.]+)', lines[i + 1])
                
                # print(schedule_time_match)
                # exit()
                if all([prefill_match, decode_match, gpu_match, cpu_match, schedule_time_match, exec_time_match]):
                    prefills.append(int(prefill_match.group(1)))
                    decodes.append(int(decode_match.group(1)))
                    total_compute.append(int(prefill_match.group(1)) + int(decode_match.group(1)))
                    gpu_usage.append(float(gpu_match.group(1)))
                    cpu_usage.append(float(cpu_match.group(1)))
                    exec_time.append(float(exec_time_match.group(1)) * 1000)  # 转换为毫秒
                    schedule_time.append(float(schedule_time_match.group(1)) * 1000)  # 转换为毫秒
                    # print(f"{int(prefill_match.group(1))},{int(decode_match.group(1))},{float(exec_time_match.group(1)) * 1000}")
            i += 2  # 每次处理两行
        else:
            i += 1

# 创建一个字典来存储不同(prefill, decode)组合的执行时间
stats = {}

# 收集数据
for p, d, t in zip(prefills, decodes, exec_time):
    key = (p, d)
    if key not in stats:
        stats[key] = []
    stats[key].append(t)

# 计算平均执行时间
avg_stats = {}
for key in stats:
    avg_stats[key] = sum(stats[key]) / len(stats[key])

# 将结果写入文件
with open(log_path + '/exec_time_stats.csv', 'w') as f:
    f.write("Prefill\tDecode\tAvg Exec Time (ms)\tCount\n")
    # 按prefill和decode数量排序
    for key in sorted(stats.keys()):
        prefill, decode = key
        avg_time = avg_stats[key]
        count = len(stats[key])
        f.write(f"{prefill},{decode},{avg_time:.2f},{count}\n")


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
plt.savefig(log_path+'/prefills.png', bbox_inches='tight')
plt.close()

# 2. Decodes图
plt.figure()
plt.plot(x, decodes, 'r-', linewidth=2)
plt.title('Number of Decodes over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Number of Decodes', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/decodes.png', bbox_inches='tight')
plt.close()

# 3. Total Compute图
plt.figure()
plt.plot(x, total_compute, 'g-', linewidth=2)
plt.title('Total Compute over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Total Compute', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/total_compute.png', bbox_inches='tight')
plt.close()

# 3. GPU Usage图
plt.figure()
plt.plot(x, gpu_usage, 'g-', linewidth=2)
plt.title('GPU Cache Usage over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('GPU Cache Usage (%)', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/gpu_usage.png', bbox_inches='tight')
plt.close()

# 4. CPU Usage图
plt.figure()
plt.plot(x, cpu_usage, 'm-', linewidth=2)
plt.title('CPU Cache Usage over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('CPU Cache Usage (%)', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/cpu_usage.png', bbox_inches='tight')
plt.close()

# GPU和CPU使用率双Y轴图
fig, ax1 = plt.subplots(figsize=(10, 6))

# GPU使用率 - 左Y轴
ax1.plot(x, gpu_usage, 'g-', linewidth=2, label='GPU Cache Usage')
ax1.set_xlabel('Schedule Call Number', fontsize=12)
ax1.set_ylabel('GPU Cache Usage (%)', fontsize=12, color='g')
ax1.tick_params(axis='y', labelcolor='g')

# CPU使用率 - 右Y轴 
ax2 = ax1.twinx()
ax2.plot(x, cpu_usage, 'm-', linewidth=2, label='CPU Cache Usage')
ax2.set_ylabel('CPU Cache Usage (%)', fontsize=12, color='m')
ax2.tick_params(axis='y', labelcolor='m')

# 添加标题和图例
plt.title('Cache Usage over Time', fontsize=14)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.grid(True)
plt.savefig(log_path+'/cache_usage_combined.png', bbox_inches='tight')
plt.close()


# 5. Execution Time图
plt.figure()
plt.plot(x, exec_time, 'y-', linewidth=2)
plt.title('Execution Time over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Execution Time (ms)', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/exec_time.png', bbox_inches='tight')
plt.close()

# 6. Schedule Time图
plt.figure()
plt.plot(x, schedule_time, 'y-', linewidth=2)
plt.title('Schedule Time over Time', fontsize=14)
plt.xlabel('Schedule Call Number', fontsize=12)
plt.ylabel('Schedule Time (ms)', fontsize=12)
plt.grid(True)
plt.savefig(log_path+'/schedule_time.png', bbox_inches='tight')
plt.close()

# 打印统计信息
print(f"总调用次数: {len(prefills)}")
print(f"Prefills - 平均: {np.mean(prefills):.2f}, 最大: {max(prefills)}, 最小: {min(prefills)}")
print(f"Decodes - 平均: {np.mean(decodes):.2f}, 最大: {max(decodes)}, 最小: {min(decodes)}")
print(f"GPU使用率 - 平均: {np.mean(gpu_usage):.2f}%, 最大: {max(gpu_usage):.2f}%, 最小: {min(gpu_usage):.2f}%")
print(f"CPU使用率 - 平均: {np.mean(cpu_usage):.2f}%, 最大: {max(cpu_usage):.2f}%, 最小: {min(cpu_usage):.2f}%")
print(f"执行时间 - 平均: {np.mean(exec_time):.2f}ms, 最大: {max(exec_time):.2f}ms, 最小: {min(exec_time):.2f}ms")
print(f"Schedule Time - 平均: {np.mean(schedule_time):.2f}ms, 最大: {max(schedule_time):.2f}ms, 最小: {min(schedule_time):.2f}ms")