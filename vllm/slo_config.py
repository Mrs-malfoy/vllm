
class SLOConfig:
    """Dataclass which contains all SLO-related configuration."""
    slo_type_num: int = 3
    ttft_slos: list[float] = [1, 1, 1]
    tbt_slos: list[float] = [1, 1, 1]

SLOConfigInstance = SLOConfig()
