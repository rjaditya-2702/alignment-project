"""schema.py — CLadder rung-2 six-step tagged schema: prompts, background synthesis,
SFT-target rendering, and the rollout parser (Variant B: model-generated mapping).

The model builds the symbol mapping from the raw description (step 0), then reasons in canonical
CLadder symbols (X=treatment, Y=outcome, V1,V2,... = the rest) so Do-Verifier parses with zero
adaptation. Output is XML-style tagged blocks, regex-parsed (not JSON — JSON hurts RL reasoning).

CLadder's own `reasoning` field is the process supervision (rendered into the SFT target by
`format_target`). Block order matches the system prompt:

    <mapping>    step0  symbol = meaning        X = rixq; Y = xevu; V1 = zuph
    <query_type>        query type              ate | collider_bias
    <graph>      step1  directed edges          V1->X,V1->Y,X->Y
    <estimand>   step2  symbolic do-form        E[Y | do(X = 1)] - E[Y | do(X = 0)]
    <data>       step4  probabilities           P(Y=1 | V1=0)=0.46  (one per line)
    <derivation> step3  estimable/identified    Σ_{V1} P(V1)[P(Y=1|V1,X=1) - P(Y=1|V1,X=0)]
    <arithmetic> step5  substitute & compute    0.79 - 0.43 = 0.36
    <answer>            Yes / No                (de-emphasized: a consequence, not a target)

CLadder ships NO edge-description prose, so `verbalize_background` synthesizes CLadder's canonical
background ("A has a direct effect on B and C.") from the gold graph + mapping in real-world terms
— this is what forces the model to do the variable-identification (mapping) work.
"""

import re

TAGS = ["mapping", "query_type", "graph", "estimand", "data", "derivation", "arithmetic", "answer"]

QUERY_TYPES = ("marginal, conditional, ate, ett, nde, nie, backdoor_adjustment_set, "
               "collider_bias, explaining_away, deterministic_counterfactual")

CLADDER_SYSTEM = f"""You are an expert in causal inference. You answer causal questions by
following a fixed six-step procedure and showing each step explicitly.

You will be given:
- A description of a hypothetical world (which variables directly affect which).
- Given numerical information (probabilities).
- A causal question.

FIRST, assign a canonical symbol to each variable in the world, following this
convention: X = the treatment/cause named in the question; Y = the outcome named
in the question; V1, V2, ... = the remaining variables (confounders, instruments,
mediators). Mark any variable stated to be unobserved.

THEN produce your answer in EXACTLY the following tagged format, with nothing
outside the tags. Use ONLY the symbols you defined in <mapping> in every
downstream expression.

<mapping>...</mapping>
<query_type>...</query_type>
<graph>...</graph>
<estimand>...</estimand>
<data>...</data>
<derivation>...</derivation>
<arithmetic>...</arithmetic>
<answer>Yes or No</answer>

Formatting rules:
- <mapping>: symbol = meaning, semicolon-separated, e.g.
  X = liking spicy food; Y = cholera; V1 = poverty (unobserved); V2 = water company
- <query_type>: one of {{{QUERY_TYPES}}}.
- <graph>: comma-separated directed edges, no spaces, e.g. V1->X,V2->X,V1->Y,X->Y
- <estimand>: the symbolic quantity the question asks for, e.g.
  E[Y | do(X = 1)] - E[Y | do(X = 0)]  or  P(Y | X)
- <data>: the probabilities available, one per line, e.g. P(X=1)=0.81
- <derivation>: the estimand rewritten into an estimable form using the graph,
  in the same expression syntax.
- <arithmetic>: substitute numbers into the derivation and compute the value.
- <answer>: Yes or No, following from the sign/magnitude of the arithmetic."""

CLADDER_USER = """Imagine a self-contained, hypothetical world with only the following
conditions, and without any unmentioned factors or causal relationships:
{background}.{unobserved_clause}

Given information: {given_info}

Question: {question}"""


# ── CauSciBench transfer eval (TEST ONLY) — potential-outcomes, method-menu, no code ────────────
# Variable roles are WITHHELD (the model must identify treatment/outcome/confounders/method-vars
# itself — that's the transfer signal). PO vocabulary, NOT CLadder's do-calculus tags. The model
# emits a method choice + variable-role slots; a fixed tool runs the estimator (model writes no code).

CAUSCI_SYSTEM = """You are an expert in causal inference for empirical research. You are given a
dataset (its description, its columns, and a causal question). Your job is to
determine HOW the causal effect should be estimated — you do NOT write code.

You reason in the potential-outcomes framework: identify the treatment, the
outcome, the confounders that must be adjusted for, and any variables specific
to the estimation method. Then you choose ONE estimation method from the fixed
list below, and you justify why the identification assumptions of that method
are satisfied by this study design.

Available estimation methods (choose EXACTLY one):
- ols            : Ordinary least squares / regression with controls. Use for
                   randomized data, or when all confounders are observed and
                   adjustment suffices.
- psm            : Propensity-score matching or IPW. Observational; adjust for
                   observed confounders by balancing treated/control groups.
- iv             : Instrumental variables. Use when there are UNOBSERVED
                   confounders and a valid instrument exists (affects treatment,
                   affects outcome only through treatment).
- did            : Difference-in-differences. Panel/repeated data; a group is
                   treated at a point in time while a control group is not;
                   relies on parallel trends.
- rdd            : Regression discontinuity. Treatment assigned by a threshold on
                   a running variable; compares units just above vs just below.
- frontdoor      : Frontdoor adjustment. Use when a mediator fully transmits the
                   treatment's effect and unobserved confounding blocks backdoor
                   adjustment.
- glm            : Generalized linear model. Non-continuous outcome (binary/count)
                   with observed confounders.

Produce your answer in EXACTLY the following tagged format, nothing outside the
tags. In the variable slots, every value must be an EXACT column name from the
dataset (or a comma-separated list of exact column names). Use "NA" for any slot
that does not apply to your chosen method.

<reasoning>
Briefly: what is the causal question asking (treatment -> outcome)? What is the
study design (randomized? observational? panel? threshold assignment?)? What
confounds the treatment-outcome relationship, and are those confounders observed
in the data? Which method's assumptions does this design satisfy, and why?
</reasoning>
<method>one of: ols | psm | iv | did | rdd | frontdoor | glm</method>
<variables>
treatment: <exact column name>
outcome: <exact column name>
confounders: <comma-separated exact column names, or NA>
instrument: <exact column name if method=iv, else NA>
running_variable: <exact column name if method=rdd, else NA>
cutoff: <numeric threshold if method=rdd, else NA>
time: <exact column name if method=did, else NA>
group: <exact column name if method=did, else NA>
mediator: <exact column name if method=frontdoor, else NA>
</variables>
<answer>
State the expected direction/sign of the effect you anticipate, in one sentence,
in terms of the treatment and outcome.
</answer>

Rules:
- Choose the method whose identification assumptions the DESIGN satisfies — do not
  default to ols out of convenience.
- Every method-specific slot required by your chosen method MUST be filled with a
  real column name. (iv needs instrument; rdd needs running_variable + cutoff;
  did needs time + group; frontdoor needs mediator.) A method without its required
  slots is an invalid answer.
- confounders = the variables you would adjust for. Include exactly those needed
  for identification — not every column, and not none."""

CAUSCI_USER = """{description}

## Dataset columns
{columns}

## Question
{question}

Follow the tagged procedure: reason about the study design and confounding, choose
one estimation method from the list, and fill in the variable roles using exact
column names. Do not write code."""


# ── background synthesis (CLadder ships no edge prose) ──────────────────────

def parse_mapping(step0: str) -> dict:
    """'Let V1 = zuph; X = rixq; Y = xevu.' -> {'V1':'zuph','X':'rixq','Y':'xevu'} (symbol->meaning)."""
    s = re.sub(r"^\s*Let\s+", "", str(step0).strip()).rstrip(".")
    m = {}
    for part in s.split(";"):
        if "=" in part:
            sym, name = part.split("=", 1)
            m[sym.strip()] = name.strip()
    return m


def _edges(graph_str: str):
    """'V1->X,V1->Y,X->Y' -> [('V1','X'),('V1','Y'),('X','Y')], insertion order preserved."""
    out = []
    for e in str(graph_str).replace("→", "->").split(","):
        if "->" in e:
            a, b = (t.strip() for t in e.split("->", 1))
            if a and b:
                out.append((a, b))
    return out


def _join(names):
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def verbalize_background(graph_str: str, mapping: dict) -> str:
    """Synthesize CLadder's canonical background from graph + mapping, in real-world terms:
    'zuph has a direct effect on rixq and xevu. rixq has a direct effect on xevu'."""
    children = {}
    order = []
    for a, b in _edges(graph_str):
        if a not in children:
            children[a] = []
            order.append(a)
        children[a].append(b)
    sents = [f"{mapping.get(a, a)} has a direct effect on {_join([mapping.get(c, c) for c in children[a]])}"
             for a in order]
    return ". ".join(sents)


# ── SFT target rendering ────────────────────────────────────────────────────

def _render_mapping(mapping: dict) -> str:
    """Canonical order X, Y, then V1,V2,... as 'X = ...; Y = ...; V1 = ...'."""
    keys = ([k for k in ("X", "Y") if k in mapping] +
            sorted((k for k in mapping if k not in ("X", "Y")), key=lambda s: (len(s), s)))
    return "; ".join(f"{k} = {mapping[k]}" for k in keys)


def format_target(reasoning: dict, meta: dict, answer: str) -> str:
    """Render a CLadder record's gold reasoning into the six-step tagged target string."""
    mapping = parse_mapping(reasoning.get("step0", ""))
    blocks = {
        "mapping": _render_mapping(mapping),
        "query_type": meta.get("query_type", ""),
        "graph": reasoning.get("step1", ""),
        "estimand": reasoning.get("step2", ""),
        "data": str(reasoning.get("step4", "")).strip(),         # keep one-per-line
        "derivation": reasoning.get("step3", ""),
        "arithmetic": reasoning.get("step5", ""),                # step5 only; `end` sign lives in <answer>
        "answer": "Yes" if str(answer).strip().lower().startswith("y") else "No",
    }
    return "\n".join(f"<{t}>{blocks[t]}</{t}>" for t in TAGS)


# ── rollout parsing ─────────────────────────────────────────────────────────
# Extract the tagged blocks from a rollout. Missing blocks are "" (not dropped) so the reward can
# gate on emptiness. Tolerant of a leading <think>...</think>, surrounding prose, and repeated tags
# (last occurrence wins — models restate the final form). answer normalized to 'yes'|'no'|''.

_PATS = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.DOTALL | re.IGNORECASE) for t in TAGS}


def parse(text: str) -> dict:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    out = {}
    for t, pat in _PATS.items():
        ms = pat.findall(text)
        out[t] = ms[-1].strip() if ms else ""
    a = out["answer"].lower()
    out["answer"] = "yes" if a.startswith("y") else "no" if a.startswith("n") else ""
    return out


def n_blocks(parsed: dict) -> int:
    """How many reasoning blocks (excludes answer) are non-empty — parse health."""
    return sum(bool(parsed[t]) for t in TAGS if t != "answer")
