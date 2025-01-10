
class SLOConfig:
    """Dataclass which contains all SLO-related configuration."""
    slo_type_num: int = 3
    ttft_slos: list[float] = [1, 1, 2]
    tbt_slos: list[float] = [0.1, 0.5, 0.1]