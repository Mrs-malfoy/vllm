import json
import os
import re
from collections import defaultdict

def calculate_goodput(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    valid_requests = 0
    completed = data['completed']
    
    for i in range(completed):
        ttfs = data['ttfss'][i]
        fit = 0.0
        if 'fit' in data:
            fit = data['fit'][i]
        if ttfs < 3 and fit == 0.0:
            valid_requests += 1
    
    goodput = valid_requests / data['duration']
    return goodput

def process_folder(folder_path):
    # 使用defaultdict来按QPS分组存储goodput
    qps_goodputs = defaultdict(list)
    
    # 获取文件夹中所有的json文件
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    for json_file in json_files:
        # 从文件名中提取QPS值
        qps_match = re.search(r'vllm-(\d+\.\d+)qps', json_file)
        if qps_match:
            qps = float(qps_match.group(1))
            file_path = os.path.join(folder_path, json_file)
            goodput = calculate_goodput(file_path)
            qps_goodputs[qps].append(goodput)
    
    # 计算每个QPS组的平均值
    qps_avg_goodputs = {}
    for qps, goodputs in sorted(qps_goodputs.items()):
        avg_goodput = sum(goodputs) / len(goodputs)
        qps_avg_goodputs[qps] = {
            'individual_goodputs': goodputs,
            'average_goodput': avg_goodput,
            'num_samples': len(goodputs)
        }
    
    return qps_avg_goodputs

# 使用示例
folder_path = "our_method"  # 替换为你的文件夹路径
results = process_folder(folder_path)

# 打印结果
for qps, data in results.items():
    print(f"\nQPS: {qps}")
    print(f"Number of samples: {data['num_samples']}")
    print(f"Individual goodputs: {data['individual_goodputs']}")
    print(f"Average goodput: {data['average_goodput']:.2f}")