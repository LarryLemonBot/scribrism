"""Frozen research-only ordered-step region candidate; never used by live review.

R3-ordered-step-256-v1 uses only cached window probabilities and token count.
It accepts no truth/order/source metadata and does not infer authorship certainty.
The two offset grids overlap and are not independent statistical observations.
"""
import math

CANDIDATE_ID = "R3-ordered-step-256-v1"
WINDOW = 256
STRIDE = 128
ANCHOR = 0.60
MIN_R2 = 0.60
MAX_LINEAR_SSE_RATIO = 0.75
MAX_BOUNDARY_DISAGREEMENT = 256
CONSTANT_EPSILON = 1e-12


def planned_windows(n_tokens):
    """Use 256-token windows at stride 128, including a final end-aligned window."""
    if type(n_tokens) is not int or n_tokens <= 0:
        raise ValueError("n_tokens must be a positive integer")
    starts = [0] if n_tokens <= WINDOW else list(range(0, n_tokens-WINDOW+1, STRIDE))
    if n_tokens > WINDOW and starts[-1] != n_tokens-WINDOW:
        starts.append(n_tokens-WINDOW)
    return [(start, min(start+WINDOW, n_tokens)) for start in starts]


def _mean(values):
    return math.fsum(values) / len(values)


def _sse(values, mean):
    return math.fsum((value-mean)**2 for value in values)


def _grid(rows, offset):
    rows = sorted((r for r in rows if r[1]-r[0] == WINDOW and r[0] % WINDOW == offset), key=lambda r: r[0])
    result = {"offset": offset, "n_windows": len(rows), "flag": False, "reason_codes": []}
    if len(rows) < 4:
        result.update(flag=None, reason_codes=["insufficient_resolution"])
        return result
    q = [p[1]+p[3]-p[0] for _, _, p in rows]
    overall = _mean(q)
    constant_sse = _sse(q, overall)
    best = min((_sse(q[:k], _mean(q[:k])) + _sse(q[k:], _mean(q[k:])), k)
               for k in range(1, len(q)))
    step_sse, cut = best
    centers = [(start+end)/2 for start, end, _ in rows]
    center_mean = _mean(centers)
    x = [value-center_mean for value in centers]
    slope = math.fsum(xx*(yy-overall) for xx, yy in zip(x, q)) / math.fsum(xx*xx for xx in x)
    linear_sse = math.fsum((yy-(overall+slope*xx))**2 for xx, yy in zip(x, q))
    left = [_mean([r[2][j] for r in rows[:cut]]) for j in range(4)]
    right = [_mean([r[2][j] for r in rows[cut:]]) for j in range(4)]
    orientation = ("HA" if left[0] >= ANCHOR and right[1]+right[3] >= ANCHOR else
                   "AH" if right[0] >= ANCHOR and left[1]+left[3] >= ANCHOR else None)
    r2 = 1-step_sse/constant_sse if constant_sse > CONSTANT_EPSILON else None
    reasons = []
    if r2 is None:
        reasons.append("near_constant_profile")
    elif r2 < MIN_R2:
        reasons.append("insufficient_step_fit")
    if step_sse > MAX_LINEAR_SSE_RATIO * linear_sse:
        reasons.append("linear_trend_not_rejected")
    if orientation is None:
        reasons.append("opposed_segment_anchors_absent")
    result.update(flag=not reasons, reason_codes=reasons, cut_index=cut,
                  boundary=rows[cut][0], order=orientation,
                  q=q, window_starts=[r[0] for r in rows],
                  constant_sse=constant_sse, step_sse=step_sse, linear_sse=linear_sse,
                  step_r2=r2, left_mean=left, right_mean=right)
    return result


def ordered_step_region(n_tokens, windows):
    """Return an experimental region flag; input windows are (start, end, p4).

    Complete finite normalized windows are required before selecting the two grids.
    Null means invalid/missing input or insufficient resolution, never Human.
    Extra mapping/label metadata has no input channel. No model or I/O is invoked.
    """
    result = {"candidate_id": CANDIDATE_ID, "status": "invalid_input", "flag": None,
              "reason_codes": [], "grids": [], "order": None, "boundary": None}
    try:
        plan = set(planned_windows(n_tokens))
    except ValueError:
        result["reason_codes"] = ["invalid_token_count"]
        return result
    if not isinstance(windows, (list, tuple)):
        result["reason_codes"] = ["invalid_windows_container"]
        return result
    rows, seen, reasons = [], set(), []
    for row in windows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            reasons.append("malformed_window")
            continue
        start, end, scores = row
        if type(start) is not int or type(end) is not int or (start, end) not in plan:
            reasons.append("window_outside_plan")
            continue
        key = (start, end)
        if key in seen:
            reasons.append("duplicate_window")
            continue
        seen.add(key)
        if (not isinstance(scores, (list, tuple)) or len(scores) != 4 or
            any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores) or
            abs(math.fsum(scores)-1) > 1e-5+1e-12):
            reasons.append("invalid_probabilities")
            continue
        rows.append((start, end, tuple(float(x) for x in scores)))
    if seen != plan:
        reasons.append("incomplete_window_plan")
    if reasons:
        result["reason_codes"] = sorted(set(reasons))
        return result
    grids = [_grid(rows, offset) for offset in (0, 128)]
    result["grids"] = grids
    if any(grid["flag"] is None for grid in grids):
        result.update(status="insufficient_resolution", reason_codes=["insufficient_resolution"])
        return result
    result["status"] = "complete"
    result["flag"] = False
    if not all(grid["flag"] for grid in grids):
        result["reason_codes"] = ["grid_gate_failed"]
        return result
    if grids[0]["order"] != grids[1]["order"]:
        result["reason_codes"] = ["grid_order_disagreement"]
        return result
    if abs(grids[0]["boundary"]-grids[1]["boundary"]) > MAX_BOUNDARY_DISAGREEMENT:
        result["reason_codes"] = ["grid_boundary_disagreement"]
        return result
    result.update(flag=True, order=grids[0]["order"],
                  boundary=(grids[0]["boundary"]+grids[1]["boundary"])/2)
    return result
