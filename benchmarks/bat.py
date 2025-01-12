import subprocess
import time

def run_benchmark(request_rate, run_number):
    cmd = f"""python benchmark_serving.py \
        --backend vllm \
        --dataset-name sharegpt \
        --dataset-path sharegpt-data.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct \
        --ignore-eos \
        --num-prompts {1600} \
        --request-rate {request_rate} \
        --save-result"""
    
    try:
        print(f"\n开始测试 request-rate = {request_rate}, 第 {run_number} 次运行")
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"request-rate {request_rate} 第 {run_number} 次测试完成")
        time.sleep(5)  # 两次测试之间暂停5秒
    except subprocess.CalledProcessError as e:
        print(f"测试失败，request-rate = {request_rate}, 第 {run_number} 次: {e}")

def main():
    # 定义要测试的request-rate值
    rates = [20]  # 可以根据需要修改
    runs_per_rate = 1  # 每个速率测试5次
    
    for rate in rates:
        for run in range(1, runs_per_rate + 1):
            run_benchmark(rate, run)
        print(f"\n完成 request-rate {rate} 的所有 {runs_per_rate} 次测试")
        time.sleep(10)  # 不同速率之间可以多休息一会

if __name__ == "__main__":
    main()