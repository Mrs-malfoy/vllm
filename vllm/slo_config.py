import os
import json
import vllm

class SLOConfig:
    """Dataclass which contains all SLO-related configuration."""
    slo_type_num: int = 3
    ttft_slos: list[float] = [1, 1, 1]
    tbt_slos: list[float] = [1, 1, 1]
    def __init__(self):

        homepath = os.path.dirname(os.path.abspath(vllm.__file__))
        file_path = os.path.join(homepath, 'envelop_config.json')
        with open(file_path, 'r') as config_file:
            config = json.load(config_file)
            # 从配置文件更新scheduler参数
            self.slo_type_num = config.get('SLO_num', 3)
            self.ttft_slos = config.get('TTFT_SLOs', [1, 1, 1])
            self.tbt_slos = config.get('TBT_SLOs', [1, 1, 1])
            print(f"SLO config updated: {self.slo_type_num}, {self.ttft_slos}, {self.tbt_slos}")

SLOConfigInstance = SLOConfig()