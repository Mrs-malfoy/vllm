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
        ttft = data['ttfts'][i]
        fit = 0.0
        if 'fit' in data:
            fit = data['fit'][i]
        if ttft <= 1.5 and fit == 0.0:
            valid_requests += 1
    
    goodput = valid_requests / data['duration']
    return goodput

goodput = calculate_goodput("/workspace/vllm/benchmarks/vllm-12.0qps-Llama-3-8B-Instruct-Chinese-20250106-080349.json")
print(goodput)