
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.sampling_params import SamplingParams
import time
from transformers import PreTrainedTokenizerBase
import numpy as np
from typing import Any, AsyncGenerator, Collection, Dict, List, Optional, Tuple
import random
import string
import matplotlib.pyplot as plt
from vllm.transformers_utils.tokenizer import get_tokenizer

# 控制显存用量一致，测试decode数量对性能的影响
def perf_dcp(model: str, block_size: int):
    # This test checks if we are able to run the engine to completion
    # without triggering asserts.
    # We are in a scenario where all blocks from the second request's prompt
    # are full and already computed when the second request arrives.
    # 生成一段长度2048的随机文本
    prompt_test = " ".join(random.choices(string.ascii_letters + string.digits, k=8000))
    print(len(prompt_test))
    engine_args = EngineArgs(model=model,
                             block_size=block_size,
                             enable_chunked_prefill=True,
                            #  tensor_parallel_size=2,
                             swap_space=16,
                            #  disable_log_requests=True,
                             preemption_mode="swap",
                             gpu_memory_utilization=0.9,
                             max_num_batched_tokens=256,
                             )

    engine = LLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams()

    fout = open('output.csv', 'a')
    # fout.write("bs,mem,time\n")


    for i in range(20):
        engine.add_request("0", prompt_test, sampling_params)
        engine.scheduler[0].scheduler_config.max_num_batched_tokens = 300000
        output = engine.step()
        engine.abort_request("0")
    node_list = []
    N = 3

    engine.scheduler[0].scheduler_config.max_num_batched_tokens = 300000

    for i in range(3, 17): # 2, 17
        bs = i * 128

        for mem in range(1, 9): # 0, 17 
            prompt_len = 16*mem

            decode_num = 128
       
            time_list = []
            for k in range(N):

                # for t in range(50):
                if(prompt_len!=0):
                    for j in range(decode_num):
                        engine.add_request(f"{j+1}", " ".join(random.choices("a", k=prompt_len)), sampling_params)
                    engine.scheduler[0].scheduler_config.max_num_batched_tokens = 300000
                    engine.step()
                engine.add_request("0", prompt_test, sampling_params)
                engine.scheduler[0].scheduler_config.max_num_batched_tokens = bs
                start_time = time.time()
                print("max_num_batched_tokens: ", engine.scheduler[0].scheduler_config.max_num_batched_tokens)
                engine.step()
                next_time = time.time() 
                print(f"Time taken: {next_time - start_time} seconds")
                time_list.append(next_time - start_time)
                fout.write(f"{bs},{prompt_len},{next_time - start_time}\n")
                fout.flush()

                engine.abort_request("0")
                for j in range(decode_num):
                    engine.abort_request(f"{j+1}")
            
            # y = sum(time_list[1:N-1]) / (N-2)
            # y = sum(time_list) / (N)
            
            # node_list.append((x, y))
    # mid_time = time.time()
    # engine.add_request("1", prompt, sampling_params)
    # engine.step()
    # end_time = time.time()
    
    # print(f"Time taken: {mid_time - start_time} seconds")
    # print(f"Time taken: {end_time - mid_time} seconds")
    # plt.xlim(left=0, right=8)
    # plt.ylim(bottom=0, top=0.3)
    # plt.plot([x[0] for x in node_list], [x[1] for x in node_list])

    # plt.savefig("perf-mem.png")


# test_flex_curve("Qwen/Qwen2.5-32B-Instruct", 16)
# perf_dcp("Qwen/Qwen2.5-32B-Instruct", 16)
perf_dcp("Rookie/Llama-3-8B-Instruct-Chinese", 16)
