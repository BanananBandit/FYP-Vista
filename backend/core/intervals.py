from typing import List, Tuple

def merge_boolean_runs(times: List[float], flags: List[bool], seg_len: float) -> List[Tuple[float, float]]:
    merged = []
    run_start = None
    for i, flag in enumerate(flags):
        t0 = times[i]
        if flag and run_start is None:
            run_start = t0
        if (not flag) and run_start is not None:
            merged.append((run_start, t0))
            run_start = None
    if run_start is not None and times:
        merged.append((run_start, times[-1] + seg_len))
    return [(max(0.0, a), max(0.0, b)) for a, b in merged if b > a]

def invert_intervals(total_start: float, total_end: float, remove: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if total_end <= total_start:
        return []

    remove_sorted = sorted((max(total_start, a), min(total_end, b)) for a, b in remove)
    remove_sorted = [(a, b) for a, b in remove_sorted if b > a]

    keep = []
    cur = total_start
    for a, b in remove_sorted:
        if a > cur:
            keep.append((cur, a))
        cur = max(cur, b)
    if cur < total_end:
        keep.append((cur, total_end))

    return [(a, b) for a, b in keep if (b - a) > 0.15]
