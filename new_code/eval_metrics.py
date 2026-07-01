"""CauSciBench eval metrics (no CLaDDer). Same metrics in SFT and RL — plotted across
both phases by plot.py.

  method_acc     fraction with correct method (bucketed like the benchmark — see _norm_method)
  treatment_acc  treatment column correct (over method-correct rows)
  outcome_acc    outcome column correct
  control_acc    coverage of reference controls recovered
  effect_acc     fraction with |pred-ref|/|ref| <= EFFECT_TOL (over runnable rows)
  mre            mean relative error of the estimated effect
"""

import config
from library import library_fn


def _norm_method(method):
    """Bucket a method name the way CauSciBench's compile_results.standardize_method_name
    does, so synonyms match: matching≡ipw≡'ps', diff_in_means≡'ols', frontdoor≡'fd', etc.
    Unrecognized names (e.g. 'backdoor') → 'other', which never matches a reference."""
    if not isinstance(method, str):
        return None
    m = method.lower()
    if "weighting" in m or "ipw" in m or "propensity" in m:                     return "ps"
    if "front" in m:                                                            return "fd"
    if "discontinuity" in m or "fuzzy" in m or "rdd" in m:                       return "rdd"
    if "in-difference" in m or "did" in m or "in-diff" in m or "fixed effects" in m or "panel" in m: return "did"
    if "matching" in m or "observational" in m:                                 return "ps"
    if "logistic" in m or "probit" in m or "logit" in m or "glm" in m:          return "glm"
    if "linear" in m or "means" in m or "ordinary" in m or "rct" in m or "ols" in m or "wls" in m:   return "ols"
    if "instrument" in m or "encouragement" in m or "2sls" in m or "iv" in m:   return "iv"
    if "null" in m or "na" in m or "n/a" in m or "none" in m:                   return None
    return "other"


def compute_eval_metrics(items):
    """items: list of (parsed_prediction, ground_truth_dict, csv_path).
    parsed may be None when the completion failed to parse. Returns {name: float}."""
    method, treat, out, cov, eff, mres = [], [], [], [], [], []

    for parsed, gt, csv_path in items:
        if parsed is None:
            method.append(0)
            continue

        pred_m, gt_m = _norm_method(parsed.get("step2")), _norm_method(gt.get("step2"))
        m_ok = int(pred_m is not None and pred_m == gt_m)
        method.append(m_ok)
        if not m_ok:
            continue

        p1, g1 = parsed.get("step1") or {}, gt.get("step1") or {}
        t_ok = int(str(p1.get("treatment", "")).strip() == str(g1.get("treatment", "")).strip())
        o_ok = int(str(p1.get("outcome", "")).strip()   == str(g1.get("outcome", "")).strip())
        treat.append(t_ok)
        out.append(o_ok)

        ref_c  = set(g1.get("controls") or [])
        pred_c = set(p1.get("controls") or [])
        cov.append(len(pred_c & ref_c) / len(ref_c) if ref_c else (1.0 if not pred_c else 0.0))

        if t_ok and o_ok and csv_path:
            parsed["step1"]["csv_path"] = csv_path
            try:
                effect, _ = library_fn(parsed)
            except Exception:
                effect = None   # malformed/insufficient spec → no effect credit, never crash the pass
            ref = gt.get("step5")
            if effect is not None and ref:
                mre = abs(effect - ref) / abs(ref)
                mres.append(mre)
                eff.append(int(mre <= config.EFFECT_TOL))

    avg = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "causci/method_acc":    avg(method),
        "causci/treatment_acc": avg(treat),
        "causci/outcome_acc":   avg(out),
        "causci/control_acc":   avg(cov),
        "causci/effect_acc":    avg(eff),
        "causci/mre":           avg(mres),
    }
