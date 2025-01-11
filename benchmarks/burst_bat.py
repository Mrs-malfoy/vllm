import subprocess
import time

def run_benchmark(scale, run_number, num_prompts):
    cmd = f"""python benchmark_serving.py\
             --backend vllm\
             --dataset-name burstgpt\
             --timestamp-file ./Burst_1day.csv\
             --model Rookie/Llama-3-8B-Instruct-Chinese\
             --ignore-eos\
             --save-result\
             --num-prompts {num_prompts}\
             --time-scale {scale}"""
    
    try:
        print(f"\n开始测试 time-scale = {scale}, 第 {run_number} 次运行")
        result = subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"time-scale {scale} 第 {run_number} 次测试完成")
        time.sleep(5)  # 两次测试之间暂停5秒
    except subprocess.CalledProcessError as e:
        print(f"测试失败，time-scale = {scale}, 第 {run_number} 次: {e}")

def main():
    # 定义要测试的request-rate值
    IRs=[3]
    origin_time = 28436

    # scales = [100, 200, 300]  # 可以根据需要修改
    runs_per_rate = 1  # 每个速率测试5次
    num_prompts = 100
    
    for ir in IRs:
        target_time = num_prompts / ir
        scale = int(origin_time / target_time)
        for run in range(1, runs_per_rate + 1):
            run_benchmark(scale, run, num_prompts)
        print(f"\n完成 ir {ir} 的所有 {runs_per_rate} 次测试")
        time.sleep(10)  # 不同速率之间可以多休息一会

if __name__ == "__main__":
    main()