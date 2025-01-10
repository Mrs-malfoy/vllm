
class SLOConfig:
    """Dataclass which contains all SLO-related configuration."""
    slo_type_num: int = 3
    ttft_slos: list[float] = [0.875, 0.450, 0.450]
    tbt_slos: list[float] = [0.450, 0.450, 0.450]