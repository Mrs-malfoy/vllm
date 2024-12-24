cd /workspace/vllm
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model Rookie/Llama-3-8B-Instruct-Chinese \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet.txt \
    --num-prompts 1000 \
    --request-rate 10