"""
Parse model completions into per-step fields.

CLadder steps: 1 (causal structure), 2 (query type), 3 (estimand derivation),
               4 (compute + code), 5 (final answer Yes/No)

CauSciBench steps: 1 (variable identification), 2 (method selection),
                   3 (estimation spec), 4 (Python code), 5 (answer)
"""

import re


# ── Generic helpers ────────────────────────────────────────────────────────────

def _extract_step(text: str, step_num: int, next_step_num: int | None = None) -> str:
    """Extract text between ## Step N and ## Step N+1 (or end of string).

    If ## Step 1 is not found (because the prompt already contained the header
    and the completion starts mid-step), step 1 content is taken from position 0
    up to the next step header.
    """
    start_pat = rf"##\s*Step\s*{step_num}\b"
    start = re.search(start_pat, text, re.IGNORECASE)
    if not start:
        if step_num == 1:
            # Prompt ended with ## Step 1 header; completion starts with step 1 body
            content_start = 0
        else:
            return ""
    else:
        content_start = start.end()
    if next_step_num is not None:
        end_pat = rf"##\s*Step\s*{next_step_num}\b"
        end = re.search(end_pat, text[content_start:], re.IGNORECASE)
        content_end = content_start + end.start() if end else len(text)
    else:
        content_end = len(text)
    return text[content_start:content_end].strip()


def _extract_code_block(text: str) -> str:
    """Extract the first ```python ... ``` block."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_answer_line(text: str) -> str:
    """Extract the last Answer: line content."""
    matches = re.findall(r"Answer\s*:\s*(.+)", text, re.IGNORECASE)
    return matches[-1].strip() if matches else ""


# ── CLadder ────────────────────────────────────────────────────────────────────

CLADDER_QUERY_TYPES = {
    "marginal", "correlation", "ate", "backadj", "det-counterfactual",
    "ett", "nde", "nie", "collider_bias", "exp_away",
}


def parse_cladder(completion: str) -> dict:
    step1 = _extract_step(completion, 1, 2)
    step2_raw = _extract_step(completion, 2, 3)
    step3 = _extract_step(completion, 3, 4)
    step4_raw = _extract_step(completion, 4, 5)
    step5_raw = _extract_step(completion, 5)

    # Step 2: normalize to known query type
    step2 = ""
    for qt in CLADDER_QUERY_TYPES:
        if re.search(rf"\b{re.escape(qt)}\b", step2_raw, re.IGNORECASE):
            step2 = qt
            break

    # Step 4: extract code block
    step4_code = _extract_code_block(step4_raw)

    # Step 5: normalize to yes/no
    # Search broadly — heading tail (": Answer") and filler before the word are stripped
    if re.search(r"\byes\b", step5_raw, re.IGNORECASE):
        step5 = "yes"
    elif re.search(r"\bno\b", step5_raw, re.IGNORECASE):
        step5 = "no"
    else:
        # fall back to Answer: line or raw prefix
        ans = _extract_answer_line(step5_raw) or step5_raw.strip()
        step5 = "yes" if ans.lower().startswith("yes") else ("no" if ans.lower().startswith("no") else ans[:10].lower())

    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4_text": step4_raw,
        "step4_code": step4_code,
        "step5": step5,
    }


# ── CauSciBench ────────────────────────────────────────────────────────────────

CAUSCI_METHODS = {
    "diff_in_means", "ols", "ipw", "matching", "did", "rdd", "iv", "frontdoor", "glm",
}


def parse_causcibench(completion: str) -> dict:
    step1 = _extract_step(completion, 1, 2)
    step2_raw = _extract_step(completion, 2, 3)
    step3 = _extract_step(completion, 3, 4)
    step4_raw = _extract_step(completion, 4, 5)
    step5_raw = _extract_step(completion, 5)

    # Step 2: normalize to known method
    step2 = ""
    for m in CAUSCI_METHODS:
        if re.search(rf"\b{re.escape(m)}\b", step2_raw, re.IGNORECASE):
            step2 = m
            break

    # Step 4: extract code block
    step4_code = _extract_code_block(step4_raw)

    # Step 5: extract numeric answer
    ans = _extract_answer_line(step5_raw) or step5_raw
    num_match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", ans)
    step5 = float(num_match.group()) if num_match else None

    return {
        "step1": step1,
        "step2": step2,
        "step3": step3,
        "step4_text": step4_raw,
        "step4_code": step4_code,
        "step5": step5,
    }


# ── Dispatch ───────────────────────────────────────────────────────────────────

def parse_completion(completion: str, source: str) -> dict:
    if source == "cladder":
        return parse_cladder(completion)
    elif source == "causcibench":
        return parse_causcibench(completion)
    raise ValueError(f"Unknown source: {source}")
