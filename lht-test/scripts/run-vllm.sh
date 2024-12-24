vllm serve Rookie/Llama-3-8B-Instruct-Chinese   --enable-chunked-prefill  --swap-space 16\
     --disable-log-requests     --preemption_mode "swap"\
       --gpu-memory-utilization 0.9 &> output.log