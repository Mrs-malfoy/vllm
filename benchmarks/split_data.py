import json
import re

def filter_conversations(input_file_path, output_file_path):
    filtered_conversations = []

    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析 JSON 行
            conversation = json.loads(line.strip())
            
            # 获取 assistant 的内容
            assistant_text = conversation.get("conversation", [{}])[0].get("assistant", "")
            
            # 过滤代码块
            if re.search(r'`', assistant_text):  # 检测代码块
                continue
            
            # 过滤表格
            if re.search(r'\|', assistant_text):  # 检测表格行
                continue
            
            # 过滤 HTML 标签
            assistant_text = re.sub(r'<.*?>', '', assistant_text)  # 移除 HTML 标签
            
            # 过滤特殊字符，只保留中文和空格
            assistant_text = re.sub(r'[^u4e00-\u9fa5\s]', '', assistant_text)  # 只保留中文和空格
            
            # 过滤空行
            if assistant_text.strip():  # 如果行不为空
                # 创建新的 JSON 对象
                filtered_conversation = {
                    "conversation_id": conversation["conversation_id"],
                    "category": conversation["category"],
                    "conversation": [{
                        "human": conversation["conversation"][0]["human"],
                        "assistant": assistant_text.strip()
                    }]
                }
                filtered_conversations.append(filtered_conversation)
    
    # 将过滤后的内容写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for conversation in filtered_conversations:
            output_file.write(json.dumps(conversation, ensure_ascii=False) + '\n')

# 输入和输出文件路径
input_file_path = './sharegpt-data.jsonl'
output_file_path = './filtered_output.jsonl'

# 过滤对话内容并输出到文件
filter_conversations(input_file_path, output_file_path)
print("success")