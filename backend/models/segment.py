from dataclasses import dataclass

@dataclass
class SegmentScore:
    t_start: float
    t_end: float
    brightness: float
    blur: float
    shake: float
    is_dark: bool
    is_blurry: bool
    is_shaky: bool
    keep: bool
    reason: str
