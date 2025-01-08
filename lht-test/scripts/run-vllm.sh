cd /workspace/vllm/
LOG_DIR=/workspace/vllm/lht-test/logs/CPtest_70B_1000_20_$(date +%Y%m%d%H%M%S)

pkill -9 -f "/usr/local/bin/vllm"
pkill -9 -f "multiprocessing.resource_tracker"
pkill -9 -f "multiprocessing.spawn"

mkdir -p $LOG_DIR
# shenzhi-wang/Llama3-70B-Chinese-Chat
# Rookie/Llama-3-8B-Instruct-Chinese
# NousResearch/Meta-Llama-3-70B-Instruct
# Qwen/Qwen2.5-32B-Instruct
MODEL=Qwen/Qwen2.5-32B-Instruct
REQUEST_RATE=20

vllm serve $MODEL  --tensor-parallel-size 2 --enable-chunked-prefill  --swap-space 16\
     --disable-log-requests     --preemption_mode "swap"\
       --gpu-memory-utilization 0.99 &> $LOG_DIR/vllm.log &

echo LOG DIR: $LOG_DIR/vllm.log
vllm_pid=$!
echo "后台命令的进程号: $vllm_pid"

counter=0
while true; do
    if ps -p $vllm_pid > /dev/null; then
        if grep -q "Avg prompt throughput" "$LOG_DIR/vllm.log"; then
            echo "vllm启动成功"
            break  # 退出循环
        else
            echo "vllm启动中，用时$counter秒"
            # if [ $counter -gt 120 ]; then
            #     echo "vllm启动失败"
            #     kill -9 $vllm_pid
            #     exit 1
            # fi
            counter=$((counter+5))
            sleep 5  # 暂停5秒后重试，避免过于频繁的检查
        fi
    else
        echo "vllm启动失败"
        exit 1
    fi
done

echo LOG INTO $LOG_DIR/bench.log
cat /workspace/vllm/lht-test/scripts/run-vllm.sh > $LOG_DIR/bench.log

python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet.txt \
    --num-prompts 300 \
    --request-rate $REQUEST_RATE &>> $LOG_DIR/bench.log

kill -9 $vllm_pid

pkill -9 -f "/usr/local/bin/vllm"
pkill -9 -f "multiprocessing.resource_tracker"
pkill -9 -f "multiprocessing.spawn"

python3 /workspace/vllm/lht-test/scripts/plot.py $LOG_DIR

echo "测试完成"
