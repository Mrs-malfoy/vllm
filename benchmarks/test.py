    
import re


def is_sentence_end(text: str) -> bool:
    is_end = text.endswith((',', '!', '?', ':', ';', '，', '。', '！', '？', '；', '：','、', '”', '“', '《', '》', '\n'))  
    if not is_end:
        print("haha")
        return False
    sentences = re.split(r'[,!?;:；。，！？：、”“《》\n]+', text.strip())
    print(f"text:{text.strip()}")
    print(f"sentences:{sentences}")
    last_sentence = sentences[-1] if sentences else ''
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', last_sentence))
    print(f"last_sentence:{last_sentence}")
    return has_chinese     # 检查文本是否以句号、逗号、感叹号、问号、分号或冒号结束
       
text = """
当然这是修订版
亲爱的Barton
希望您一切都好。在我们努力创造更加分散的能源未来的过程中重要的是我们采取协作和多元化的方法来解决问题。我们相信Plurigrid的成功不仅在于您的远见卓识还在于团队的集体行动和协调。
正如分散式能源网络的控制回路需要不断的反馈和调整以保持稳定一样我们的团队动态也需要开放的沟通和透明度。"""
print(is_sentence_end(text))