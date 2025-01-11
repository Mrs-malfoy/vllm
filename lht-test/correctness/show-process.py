import matplotlib.pyplot as plt
import re

def parse_log_file(log_lines):
    stats = {
        'decode': [],
        'prefill': [],
        'total_new_token': [],
        'swap': [],
        'exec_time': []
    }
    timestamps = []
    preempt_positions = []
    
    line_number = 0
    for line in log_lines:
        line_number += 1
        
        # 解析统计数据
        if "Schedule stats" in line:
            match = re.search(r'decode:(\d+), prefill:(\d+), total_new_token:(\d+), swap:(\d+).*exec_time:([0-9.]+)', line)
            if match:
                stats['decode'].append(int(match.group(1)))
                stats['prefill'].append(int(match.group(2)))
                stats['total_new_token'].append(int(match.group(3)))
                stats['swap'].append(int(match.group(4)))
                stats['exec_time'].append(float(match.group(5)))
                timestamps.append(line_number)
        
        # 检测强制抢占成功的位置
        if "force preempt success" in line:
            # 记录下一个数据点的位置
            preempt_positions.append(line_number)
    
    return stats, timestamps, preempt_positions

def plot_stats(stats, timestamps, preempt_positions):
    # 创建5个子图
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
    
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
    
    # 第三个子图: swap
    ax3.plot(timestamps, stats['swap'], label='swap', marker='.', color='red')
    ax3.set_title('Swap')
    ax3.legend()
    ax3.grid(True)
    
    # 第四个子图: total_new_token
    ax4.plot(timestamps, stats['total_new_token'], label='total_new_token', marker='.', color='purple')
    ax4.set_title('Total New Token')
    ax4.legend()
    ax4.grid(True)
    
    # 第五个子图: exec_time
    ax5.plot(timestamps, stats['exec_time'], label='exec_time', marker='.', color='orange')
    ax5.set_title('Execution Time')
    ax5.set_xlabel('Log Line Number')
    ax5.legend()
    ax5.grid(True)
    
    # 在所有子图中添加抢占位置的竖虚线
    for pos in preempt_positions:
        next_timestamp = min([t for t in timestamps if t > pos], default=None)
        if next_timestamp:
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.axvline(x=next_timestamp, color='red', linestyle='--', alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    # plt.show()
    plt.savefig('vllm-dcp.png')
# 读取日志文件
with open('/workspace/vllm/lht-test/correctness/vllm-dcp.log', 'r') as f:
    log_lines = f.readlines()

# 解析并绘图
stats, timestamps, preempt_positions = parse_log_file(log_lines)
plot_stats(stats, timestamps, preempt_positions)