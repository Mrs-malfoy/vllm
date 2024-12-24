import json

# 读取 JSON 文件
with open('benchmarks/vllm-9.5qps-Llama-3-8B-Instruct-Chinese-20241217-223213.json', 'r') as f:
    data = json.load(f)

# 获取 ttfs > 10 的索引
high_ttfs_indices = [i for i, ttfs in enumerate(data['ttfss']) if ttfs > 10]

print(f"找到 {len(high_ttfs_indices)} 个 ttfs > 10 的请求")

# 对每个索引，输出相关信息
for idx in high_ttfs_indices[:10]:  # 只显示前10个以避免输出过多
    print(f"\n请求索引: {idx}")
    print(f"TTFS: {data['ttfss'][idx]:.2f}秒")
    print(f"TTFT: {data['ttfts'][idx]:.2f}秒")
    print(f"前20个输入token长度: {data['itls'][idx][:20]}")
    print(f"前一百个字符是: {data['generated_texts'][idx][:100]}")
