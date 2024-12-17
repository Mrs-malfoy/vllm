#!/bin/bash

# 删除所有未提交的更改
clean_changes() {
    echo "Cleaning uncommitted changes..."
    git checkout -- .
    git clean -fd
}

run_benchmark() {
    # 启动vllm服务在后台运行，并将输出重定向到nohup.out
    echo "Starting vllm service..."
    nohup vllm serve Rookie/Llama-3-8B-Instruct-Chinese \
        --swap-space 64 \
        --disable-log-requests \
        --preemption_mode "swap" \
        --gpu-memory-utilization 0.9 > vllm.log 2>&1 &
    
    # 保存进程ID
    VLLM_PID=$!
    
    # 等待服务启动（检查日志文件）
    echo "Waiting for service to start..."
    while ! grep -q "Uvicorn running on" vllm.log; do
        sleep 1
    done
    
    # 额外等待确保服务完全启动
    sleep 10

    cd benchmarks    
    echo "Running benchmark..."
    # 运行benchmark
    python bat.py
    cd ..

    echo "Stopping vllm service..."
    # 终止vllm服务
    kill $VLLM_PID
    
    # 等待进程完全终止
    wait $VLLM_PID 2>/dev/null
    
    # 清理日志文件
    rm vllm.log
}


# 运行主分支的测试
echo "Running benchmark on current branch..."
run_benchmark

# 清理更改并切换到feature分支
clean_changes
echo "Switching to feature/scheduler-algorithm branch..."
git checkout feature/scheduler-algorithm

# 清理feature分支的更改
clean_changes

# 运行feature分支的测试
echo "Running benchmark on feature/scheduler-algorithm branch..."
run_benchmark

echo "All benchmarks completed!"
