import re
import io
import contextlib

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")


def _sanitize_col(name: str) -> str:
    """Replace characters patsy treats as operators (dots, spaces, hyphens) with underscores."""
    return re.sub(r'[.\s\-]', '_', str(name))

# @lru_cache(maxsize=64)
def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")
    # Sanitize column names so patsy formulas never see dots/spaces/hyphens
    df.columns = [_sanitize_col(c) for c in df.columns]
    return df


def _formula(outcome, treatment, controls):
    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
    ]
    rhs = [treatment] + clean_controls
    return f"{outcome} ~ {' + '.join(rhs)}"


def _result(coef, se, alpha=0.05):
    from scipy import stats
    t_stat = coef / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=1000))
    return {
        "effect": float(coef),
        "se": float(se),
        "is_significant": bool(p_value < alpha),
    }


# ------------------------------------------------------------------
# 1. diff_in_means
# ------------------------------------------------------------------
def run_diff_in_means(df, treatment, outcome, controls=None):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2:
        return _result(0.0, 0.0)
    formula = _formula(outcome, treatment, clean_controls)
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    if treatment not in model.params:
        return _result(0.0, 0.0)
    coef = model.params[treatment]
    se   = model.bse[treatment]
    return _result(coef, se)


# ------------------------------------------------------------------
# 2. ols
# ------------------------------------------------------------------
def run_ols(df, treatment, outcome, controls=None):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2:
        return _result(0.0, 0.0)
    formula = _formula(outcome, treatment, clean_controls)
    model = smf.ols(formula, data=df).fit(cov_type="HC3")
    if treatment not in model.params:
        return _result(0.0, 0.0)
    coef = model.params[treatment]
    se   = model.bse[treatment]
    return _result(coef, se)


# ------------------------------------------------------------------
# 3. ipw
# ------------------------------------------------------------------
def run_ipw(df, treatment, outcome, controls=None, estimand="ATE"):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2 or df[treatment].nunique() < 2:
        return _result(0.0, 0.0)
    X = df[clean_controls].values if clean_controls else np.ones((len(df), 1))
    T = df[treatment].values
    Y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1]

    # clip propensity scores for stability
    ps = np.clip(ps, 0.01, 0.99)

    if estimand == "ATE":
        w1 = T / ps
        w0 = (1 - T) / (1 - ps)
        denom1, denom0 = np.mean(w1), np.mean(w0)
        if denom1 == 0 or denom0 == 0:
            return _result(0.0, 0.0)
        tau = np.mean(Y * w1) / denom1 - np.mean(Y * w0) / denom0
    elif estimand == "ATT":
        w0 = ps / (1 - ps)
        treated = Y[T == 1]
        denom0 = np.sum(w0[T == 0])
        if len(treated) == 0 or denom0 == 0:
            return _result(0.0, 0.0)
        tau = np.mean(treated) - np.sum(Y[T == 0] * w0[T == 0]) / denom0
    else:  # ATC
        w1 = (1 - ps) / ps
        denom1 = np.sum(w1[T == 1])
        if len(Y[T == 0]) == 0 or denom1 == 0:
            return _result(0.0, 0.0)
        tau = np.sum(Y[T == 1] * w1[T == 1]) / denom1 - np.mean(Y[T == 0])

    # bootstrap SE
    n_boot = 200
    boot_taus = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        Xb, Tb, Yb, psb = X_scaled[idx], T[idx], Y[idx], ps[idx]
        psb = np.clip(psb, 0.01, 0.99)
        if estimand == "ATE":
            w1b = Tb / psb
            w0b = (1 - Tb) / (1 - psb)
            boot_taus.append(
                np.mean(Yb * w1b) / np.mean(w1b) - np.mean(Yb * w0b) / np.mean(w0b)
            )
        else:
            boot_taus.append(tau)  # fallback
    se = float(np.std(boot_taus))
    return _result(tau, se)


# ------------------------------------------------------------------
# 4. matching
# ------------------------------------------------------------------
def run_matching(df, treatment, outcome, controls=None, estimand="ATT"):
    from sklearn.neighbors import NearestNeighbors

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2 or df[treatment].nunique() < 2:
        return _result(0.0, 0.0)
    X = df[clean_controls].values if clean_controls else np.ones((len(df), 1))
    T = df[treatment].values
    Y = df[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # fit propensity score
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_scaled, T)
    ps = lr.predict_proba(X_scaled)[:, 1].reshape(-1, 1)

    treated_idx   = np.where(T == 1)[0]
    control_idx   = np.where(T == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return _result(0.0, 0.0)

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(ps[control_idx])
    _, indices = nn.kneighbors(ps[treated_idx])
    matched_control_idx = control_idx[indices.flatten()]

    if estimand == "ATT":
        tau = np.mean(Y[treated_idx] - Y[matched_control_idx])
    elif estimand == "ATC":
        # Match control units to treated
        nn2 = NearestNeighbors(n_neighbors=1)
        nn2.fit(ps[treated_idx])

        _, indices2 = nn2.kneighbors(ps[control_idx])
        matched_treated_idx = treated_idx[indices2.flatten()]

        tau = np.mean(Y[matched_treated_idx] - Y[control_idx])

    else:  # ATE
        # ATT component
        att = np.mean(Y[treated_idx] - Y[matched_control_idx])

        # ATC component
        nn2 = NearestNeighbors(n_neighbors=1)
        nn2.fit(ps[treated_idx])

        _, indices2 = nn2.kneighbors(ps[control_idx])
        matched_treated_idx = treated_idx[indices2.flatten()]

        atc = np.mean(Y[matched_treated_idx] - Y[control_idx])

        # Weighted average
        tau = (
            len(treated_idx) * att +
            len(control_idx) * atc
        ) / len(T)
    # elif estimand == "ATE":  # ATE
    #     nn2 = NearestNeighbors(n_neighbors=1)
    #     nn2.fit(ps[treated_idx])
    #     _, indices2 = nn2.kneighbors(ps[control_idx])
    #     matched_treated_idx = treated_idx[indices2.flatten()]
    #     att = np.mean(Y[treated_idx] - Y[matched_control_idx])
    #     atc = np.mean(Y[matched_treated_idx] - Y[control_idx])
    #     tau = (len(treated_idx) * att + len(control_idx) * atc) / len(T)

    # bootstrap SE
    n_boot = 200
    boot_taus = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        Xb, Tb, Yb = X_scaled[idx], T[idx], Y[idx]
        psb = lr.predict_proba(Xb)[:, 1].reshape(-1, 1)
        ti = np.where(Tb == 1)[0]
        ci = np.where(Tb == 0)[0]
        if len(ti) == 0 or len(ci) == 0:
            continue
        nn_b = NearestNeighbors(n_neighbors=1).fit(psb[ci])
        _, mi = nn_b.kneighbors(psb[ti])
        boot_taus.append(float(np.mean(Yb[ti] - Yb[ci[mi.flatten()]])))
    se = float(np.std(boot_taus)) if boot_taus else 0.0
    return _result(tau, se)


# ------------------------------------------------------------------
# 5. did
# ------------------------------------------------------------------
def run_did(df, treatment, outcome, time_variable, group_variable, controls=None):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy()

    # detect canonical vs TWFE
    n_time_periods = df[time_variable].nunique()

    if n_time_periods == 2:
        # canonical 2x2 DiD
        post  = time_variable
        treat = group_variable
        interaction = f"{post}_x_{treat}"
        df[interaction] = df[post] * df[treat]
        # Explicit dropna so df[group_variable] stays in sync with smf's design matrix
        df = df.dropna(subset=[outcome, post, treat, interaction] + clean_controls)
        if len(df) < 2:
            return _result(0.0, 0.0)
        rhs = [post, treat, interaction] + clean_controls
        formula = f"{outcome} ~ {' + '.join(rhs)}"
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[group_variable]}
        )
        if interaction not in model.params:
            return _result(0.0, 0.0)
        coef = model.params[interaction]
        se   = model.bse[interaction]
    else:
        # TWFE — staggered treatment
        df = df.dropna(subset=[treatment, outcome, group_variable, time_variable] + clean_controls)
        controls = [c for c in clean_controls if c != treatment]
        controls = list(dict.fromkeys(controls))  # dedupe, preserve order
        if len(df) < 2:
            return _result(0.0, 0.0)
        df["unit_id"] = pd.Categorical(df[group_variable]).codes
        df["time_id"] = pd.Categorical(df[time_variable]).codes
        df = df.set_index(["unit_id", "time_id"])

        exog_cols = [treatment] + (controls or [])
        exog = df[exog_cols]
        # PanelOLS.has_constant fails when >1 column is all-constant; drop them
        exog = exog.loc[:, exog.apply(lambda c: c.nunique() > 1)]
        if treatment not in exog.columns:
            return _result(0.0, 0.0)
        model = PanelOLS(
            df[outcome],
            exog,
            entity_effects=True,
            time_effects=True,
            check_rank=False,
            drop_absorbed=True,
        ).fit(cov_type="clustered", cluster_entity=True)
        if treatment not in model.params.index:
            return _result(0.0, 0.0)
        coef = float(model.params[treatment])
        se   = float(model.std_errors[treatment])

    return _result(coef, se)


# ------------------------------------------------------------------
# 6. rdd
# ------------------------------------------------------------------
def run_rdd(df, treatment, outcome, running_variable, cutoff, controls=None):
    from rdd import rdd

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(
        subset=[running_variable, outcome] + clean_controls
    )

    if len(df) < 2:
        return _result(0.0, 0.0)
    # rdd package expects the running variable centered at cutoff
    df["_running"] = df[running_variable] - cutoff

    model = rdd.rdd(
        input_data=df,
        xname="_running",
        yname=outcome,
        cut=0.0,
        controls=clean_controls,
        verbose=False,
    )
    result = model.fit()
    if "TREATED" not in result.params:
        return _result(0.0, 0.0)
    coef = result.params["TREATED"]
    se   = result.bse["TREATED"]
    return _result(coef, se)


# ------------------------------------------------------------------
# 7. iv (2SLS)
# ------------------------------------------------------------------
def run_iv(df, treatment, outcome, instrument, controls=None):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(subset=[treatment, outcome, instrument] + clean_controls)
    if len(df) < 2:
        return _result(0.0, 0.0)
    if df[treatment].nunique() < 2 or df[instrument].nunique() < 2:
        return _result(0.0, 0.0)

    df["const"] = 1.0
    # Drop zero-variance controls — IV2SLS does a strict full-rank check on [exog, endog]
    valid_controls = [c for c in clean_controls if df[c].nunique() > 1]
    exog_cols = ["const"] + valid_controls

    # Pre-check both rank conditions that IV2SLS._validate_inputs() enforces
    try:
        mat_endog = np.column_stack([df[exog_cols].values, df[[treatment]].values]).astype(float)
        mat_instr = np.column_stack([df[exog_cols].values, df[[instrument]].values]).astype(float)
    except (ValueError, TypeError):
        return _result(0.0, 0.0)
    if np.linalg.matrix_rank(mat_endog) < mat_endog.shape[1]:
        return _result(0.0, 0.0)
    if np.linalg.matrix_rank(mat_instr) < mat_instr.shape[1]:
        return _result(0.0, 0.0)

    model = IV2SLS(
        dependent=df[outcome],
        exog=df[exog_cols],
        endog=df[treatment],
        instruments=df[instrument],
    ).fit(cov_type="robust")

    coef = float(model.params[treatment])
    se   = float(model.std_errors[treatment])
    return _result(coef, se)


# ------------------------------------------------------------------
# 8. frontdoor
# ------------------------------------------------------------------
def run_frontdoor(df, treatment, outcome, mediator, controls=None):
    """
    Frontdoor adjustment:
    P(Y|do(T)) = Σ_m P(M=m|T) * Σ_t P(T=t) * P(Y|M=m, T=t)
    
    Implemented as two-stage regression with bootstrap SE.
    Stage 1: M ~ T (+ controls)
    Stage 2: Y ~ M + T (+ controls), using stage 1 predictions
    ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
    """

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(
        subset=[treatment, outcome, mediator] + clean_controls
    )
    if len(df) < 2:
        return _result(0.0, 0.0)

    def _frontdoor_ate(df):
        # stage 1: T -> M
        f1 = _formula(mediator, treatment, clean_controls)
        m1 = smf.ols(f1, data=df).fit()
        df = df.copy()
        df["_m_hat_t1"] = m1.predict(df.assign(**{treatment: 1}))
        df["_m_hat_t0"] = m1.predict(df.assign(**{treatment: 0}))

        # stage 2: (M, T) -> Y
        f2 = _formula(outcome, mediator, [treatment] + clean_controls)
        m2 = smf.ols(f2, data=df).fit()

        # ATE via frontdoor formula
        y1 = m2.predict(df.assign(**{mediator: df["_m_hat_t1"], treatment: 1}))
        y0 = m2.predict(df.assign(**{mediator: df["_m_hat_t0"], treatment: 0}))
        return float(np.mean(y1 - y0))

    tau = _frontdoor_ate(df)

    # bootstrap SE
    n_boot = 200
    boot_taus = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        boot_df = df.sample(frac=1, replace=True, random_state=int(rng.integers(0, 9999)))
        try:
            boot_taus.append(_frontdoor_ate(boot_df))
        except Exception:
            continue
    se = float(np.std(boot_taus)) if boot_taus else 0.0
    return _result(tau, se)


# ------------------------------------------------------------------
# 9. glm
# ------------------------------------------------------------------

def run_glm(df, treatment, outcome, controls=None):

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2:
        return _result(0.0, 0.0)
    formula = _formula(outcome, treatment, clean_controls)

    # detect outcome type
    unique_vals = df[outcome].dropna().unique()
    is_binary = set(unique_vals).issubset({0, 1, 0.0, 1.0})
    is_count  = (
        not is_binary
        and np.issubdtype(df[outcome].dtype, np.integer)
        and df[outcome].min() >= 0
    )

    if is_binary:
        if len(unique_vals) < 2:
            return _result(0.0, 0.0)
        try:
            model = smf.logit(formula, data=df).fit(disp=False)
        except np.linalg.LinAlgError:
            model = smf.ols(formula, data=df).fit(cov_type="HC3")
    elif is_count:
        try:
            model = smf.poisson(formula, data=df).fit(disp=False)
        except np.linalg.LinAlgError:
            model = smf.ols(formula, data=df).fit(cov_type="HC3")
    else:
        # fallback to OLS if outcome doesn't fit binary or count
        model = smf.ols(formula, data=df).fit(cov_type="HC3")

    if treatment not in model.params:
        return _result(0.0, 0.0)
    coef = model.params[treatment]
    se   = model.bse[treatment]
    return _result(coef, se)

# ------------------------------------------------------------------
# 10. backdoor (g-computation / standardization)
# ------------------------------------------------------------------
def run_backdoor(df, treatment, outcome, controls=None, estimand="ATE"):
    """
    Backdoor adjustment via g-computation (standardization).

    Given a sufficient adjustment set Z (the `controls`), estimate:
        E[Y | do(T=t)] = E_Z[ E[Y | T=t, Z] ]

    ATE = E[Y | do(T=1)] - E[Y | do(T=0)]
    ATT = E[Y | do(T=1), T=1] - E[Y | do(T=0), T=1]
    ATC = E[Y | do(T=1), T=0] - E[Y | do(T=0), T=0]

    The adjustment set is assumed valid (LLM-specified via `controls`).
    Estimation uses a flexible outcome regression with T:Z interactions
    so the conditional outcome surface can vary across treatment arms.
    """

    clean_controls = [
        c for c in (controls or [])
        if c and isinstance(c, str) and c.strip()
        and c.strip() in df.columns  # must actually exist in the dataframe
    ]

    df = df.copy().dropna(subset=[treatment, outcome] + clean_controls)
    if len(df) < 2 or df[treatment].nunique() < 2:
        return _result(0.0, 0.0)

    # Detect outcome type for the outcome model
    unique_vals = df[outcome].dropna().unique()
    is_binary_outcome = set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) >= 2

    # Build outcome model: Y ~ T + Z + T:Z   (interactions let CATE vary in Z)
    if clean_controls:
        interaction_terms = " + ".join([f"{treatment}:{c}" for c in clean_controls])
        rhs = f"{treatment} + {' + '.join(clean_controls)} + {interaction_terms}"
    else:
        rhs = treatment
    formula = f"{outcome} ~ {rhs}"

    def _g_compute(data):
        # Fit outcome model
        try:
            if is_binary_outcome:
                model = smf.logit(formula, data=data).fit(disp=False)
            else:
                model = smf.ols(formula, data=data).fit()
        except (np.linalg.LinAlgError, ValueError):
            # fall back to additive (no interactions) on rank failure
            rhs_fallback = _formula(outcome, treatment, clean_controls)
            try:
                if is_binary_outcome:
                    model = smf.logit(rhs_fallback, data=data).fit(disp=False)
                else:
                    model = smf.ols(rhs_fallback, data=data).fit()
            except (np.linalg.LinAlgError, ValueError):
                return None

        # Choose the population to standardize over based on estimand
        if estimand == "ATT":
            pop = data[data[treatment] == 1]
        elif estimand == "ATC":
            pop = data[data[treatment] == 0]
        else:  # ATE
            pop = data

        if len(pop) == 0:
            return None

        # Counterfactual predictions: set T=1 for everyone, then T=0 for everyone
        pop_t1 = pop.assign(**{treatment: 1})
        pop_t0 = pop.assign(**{treatment: 0})

        try:
            y1 = model.predict(pop_t1)
            y0 = model.predict(pop_t0)
        except Exception:
            return None

        return float(np.mean(y1 - y0))

    tau = _g_compute(df)
    if tau is None:
        return _result(0.0, 0.0)

    # Bootstrap SE
    n_boot = 200
    boot_taus = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        boot_df = df.iloc[idx].reset_index(drop=True)
        b = _g_compute(boot_df)
        if b is not None and np.isfinite(b):
            boot_taus.append(b)
    se = float(np.std(boot_taus)) if boot_taus else 0.0

    return _result(tau, se)

# ------------------------------------------------------------------
# Dispatcher
# ------------------------------------------------------------------
ESTIMATORS = {
    "diff_in_means": run_diff_in_means,
    "ols":           run_ols,
    "ipw":           run_ipw,
    "matching":      run_matching,
    "did":           run_did,
    "rdd":           run_rdd,
    "iv":            run_iv,
    "frontdoor":     run_frontdoor,
    "backdoor":      run_backdoor,
    "glm":           run_glm,
}

def library_fn(parsed_prediction: dict) -> (float, bool):
    """
    Receives parsed CauSciBench prediction, loads CSV, runs the
    appropriate estimator, returns the effect as a float.
    Returns 0.0 on any failure.
    """

    s1     = parsed_prediction["step1"]
    method = parsed_prediction["step2"]

    df = _load(s1["csv_path"])

    # _load sanitizes column names (dots/spaces → underscores); mirror that here
    treatment = _sanitize_col(s1["treatment"])
    outcome   = _sanitize_col(s1["outcome"])
    controls  = [_sanitize_col(c) for c in (s1.get("controls") or [])]

    if method in ("diff_in_means", "ols", "glm"):
        result = ESTIMATORS[method](df, treatment, outcome, controls)

    elif method == "ipw":
        estimand = s1.get("estimand", "ATE").upper()
        result   = run_ipw(df, treatment, outcome, controls, estimand)

    elif method == "matching":
        estimand = s1.get("estimand", "ATT").upper()
        result   = run_matching(df, treatment, outcome, controls, estimand)

    elif method == "did":
        result = run_did(
            df, treatment, outcome,
            _sanitize_col(s1["time_variable"]), _sanitize_col(s1["group_variable"]), controls
        )

    elif method == "rdd":
        try:
            cutoff = float(s1["cutoff"])
        except (ValueError, TypeError):
            return 0.0, False
        result = run_rdd(
            df, treatment, outcome,
            _sanitize_col(s1["running_variable"]), cutoff, controls
        )

    elif method == "iv":
        result = run_iv(df, treatment, outcome, _sanitize_col(s1["instrument"]), controls)

    elif method == "frontdoor":
        result = run_frontdoor(df, treatment, outcome, _sanitize_col(s1["mediator"]), controls)

    elif method == "backdoor":
        estimand = s1.get("estimand", "ATE").upper()
        result = run_backdoor(df, treatment, outcome, controls, estimand)

    else:
        return 0.0, False

    return result["effect"], True