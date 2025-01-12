import matplotlib.pyplot as plt
import re

def parse_log_file(log_lines):
    stats = {
        'decode': [],
        'prefill': [],
        'total_new_token': [],
        'swap': [],
        'swapped': [],
        'exec_time': [],
        'g_mem_use': []
    }
    timestamps = []
    preempt_positions = []
    preempt_fail_positions = []
    line_number = 0
    for line in log_lines:
        
        
        # 解析统计数据
        if "Schedule stats" in line:
            line_number += 1
            match = re.search(r'decode:(\d+), prefill:(\d+), swapped:(\d+), total_new_token:(\d+), swap:(\d+).*exec_time:([0-9.]+).*g_mem_use:([0-9.]+)', line)
            if match:
                stats['decode'].append(int(match.group(1)))
                stats['prefill'].append(int(match.group(2)))
                stats['swapped'].append(int(match.group(3)))
                stats['total_new_token'].append(int(match.group(4)))
                stats['swap'].append(int(match.group(5)))
                stats['exec_time'].append(float(match.group(6)))
                stats['g_mem_use'].append(float(match.group(7)))
                timestamps.append(line_number)
        
        # 检测强制抢占成功的位置
        if "force preempt success" in line:
            # 记录下一个数据点的位置
            preempt_positions.append(line_number)

        if "force preempt not ok" in line:
            preempt_fail_positions.append(line_number)
    print(preempt_fail_positions)
    
    return stats, timestamps, preempt_positions, preempt_fail_positions

def plot_stats(stats, timestamps, preempt_positions, preempt_fail_positions):
    # 创建5个子图
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(15, 20), sharex=True)
    
    # 第一个子图: decode
    ax1.plot(timestamps, stats['decode'], label='decode', marker='.', color='blue')
    ax1.set_title('Decode')
    ax1.legend()
    ax1.grid(True)
    
    # 第二个子图: prefill
    ax2.plot(timestamps, stats['prefill'], label='prefill', marker='.', color='green')
    ax2.set_title('Prefill')
    ax2.legend()
    ax2.grid(True)

    # 第三个子图: swapped
    ax3.plot(timestamps, stats['swapped'], label='swapped', marker='.', color='red')
    ax3.set_title('Swapped')
    ax3.legend()
    ax3.grid(True)
    
    # 第四个子图: swap
    ax4.plot(timestamps, stats['swap'], label='swap', marker='.', color='red')
    ax4.set_title('Swap')
    ax4.legend()
    ax4.grid(True)
    
    # 第五个子图: total_new_token
    ax5.plot(timestamps, stats['total_new_token'], label='total_new_token', marker='.', color='purple')
    ax5.set_title('Total New Token')
    ax5.legend()
    ax5.grid(True)
    
    # 第五个子图: exec_time
    ax6.plot(timestamps, stats['exec_time'], label='exec_time', marker='.', color='orange')
    ax6.set_title('Execution Time')
    ax6.set_xlabel('Log Line Number')
    ax6.legend()
    ax6.grid(True)
    
    # 第六个子图: g_mem_use
    ax7.plot(timestamps, stats['g_mem_use'], label='g_mem_use', marker='.', color='purple')
    ax7.set_title('GPU Memory Utilization')
    ax7.set_xlabel('Log Line Number')
    ax7.legend()
    ax7.grid(True)
    
    # 在所有子图中添加抢占位置的竖虚线
    for pos in preempt_positions:
        next_timestamp = min([t for t in timestamps if t > pos], default=None)
        if next_timestamp:
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.axvline(x=next_timestamp, color='blue', linestyle='--', alpha=0.1)
    
    for pos in preempt_fail_positions:
        next_timestamp = min([t for t in timestamps if t > pos], default=None)
        if next_timestamp:
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.axvline(x=next_timestamp, color='yellow', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    # plt.show()
    plt.savefig('vllm-dcp.png')
# 读取日志文件
with open('/workspace/vllm/lht-test/correctness/vllm-dcp.log', 'r') as f:
    log_lines = f.readlines()

# 解析并绘图
stats, timestamps, preempt_positions, preempt_fail_positions = parse_log_file(log_lines)
plot_stats(stats, timestamps, preempt_positions, preempt_fail_positions)