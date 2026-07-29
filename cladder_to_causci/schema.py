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

CAUSCI_SYSTEM = """You are an applied econometrician. Given a dataset description, its columns, and a causal question, you decide HOW the causal effect should be estimated. You do NOT write code and you do NOT compute the number — a separate tool does that. Reason in the potential-outcomes framework.

Choose EXACTLY ONE method from this fixed menu (do not invent others):
  ols        — OLS / difference-in-means / regression with controls
  psm        — propensity-score matching or IPW
  iv         — instrumental variables (2SLS)
  did        — difference-in-differences
  rdd        — regression discontinuity
  frontdoor  — frontdoor adjustment
  glm        — generalized linear model (non-continuous outcome)

Carry out your reasoning DIRECTLY inside the tags below, in this exact order, with nothing outside the tags. The tags are not a separate reporting step — they are where you do the work.

<cues>
Read the description and columns. For EACH cue answer YES / NO / UNCLEAR and quote the exact sentence(s) that justify it. If nothing in the description supports a cue, it is NO or UNCLEAR — never infer it.
randomized: YES/NO/UNCLEAR — "quote"          (was treatment RANDOMLY assigned / an RCT?)
threshold: YES/NO/UNCLEAR — "quote"           (is treatment assigned by a cutoff on a continuous running variable?)
panel: YES/NO/UNCLEAR — "quote"               (repeated pre/post periods with a treated group and an untreated control group?)
instrument: YES/NO/UNCLEAR — "quote"          (a variable affecting treatment but the outcome ONLY through treatment, else as-good-as-random?)
mediator_unobserved: YES/NO/UNCLEAR — "quote" (a mediator fully on the treatment->outcome path WHILE an important confounder is unobserved?)
confounders_observed: YES/NO/UNCLEAR — "quote" (observational, common causes MEASURED in the columns, with overlap?)
</cues>
<method_reasoning>
Map cues to a method, applying these rules in order and stopping at the first match:
  1. randomized = YES            -> ols  (difference-in-means; controls only for precision)
  2. threshold  = YES            -> rdd
  3. panel      = YES            -> did
  4. instrument = YES            -> iv
  5. mediator_unobserved = YES   -> frontdoor
  6. confounders_observed = YES  -> psm  (or ols if outcome continuous and linearity reasonable; glm if outcome binary/count)
  7. none cleanly holds          -> ols, and state explicitly that this assumes unconfoundedness the data may not support (do not silently default).
State the ONE chosen method, the single identifying assumption it requires, and which cue(s) satisfy it. Do not default to ols out of convenience — rule it out unless randomized=YES or no other cue holds.
</method_reasoning>
<method>one of: ols | psm | iv | did | rdd | frontdoor | glm</method>
<variable_typing>
Fill this block ONLY if method is ols, psm, or glm; otherwise write NA.
For EACH candidate column, give exactly one label with a one-line causal reason:
  CONFOUNDER  — causes BOTH treatment and outcome; measured BEFORE treatment
  MEDIATOR    — lies on treatment -> ... -> outcome                 -> EXCLUDE
  COLLIDER    — caused by both treatment and outcome (or their effects) -> EXCLUDE
  INSTRUMENT  — affects treatment, affects outcome only via treatment -> EXCLUDE (belongs in an IV design)
  POST_TREAT  — realized after / affected by treatment              -> EXCLUDE
  IRRELEVANT  — not a cause of treatment or outcome                 -> EXCLUDE
The adjustment set = ONLY the columns labeled CONFOUNDER. Pick the SMALLEST set that blocks confounding; do not add variables "just in case." If a necessary confounder is not among the columns (unobserved), say so — ols/psm may be the wrong method and iv or frontdoor may be needed; if so, revise <method> above.
</variable_typing>
<variables>
treatment: <exact column name>
outcome: <exact column name>
confounders: <comma-separated CONFOUNDER columns if method in {ols,psm,glm}, else NA>
instrument: <column if method=iv, else NA>
running_variable: <column if method=rdd, else NA>
cutoff: <numeric threshold if method=rdd, else NA>
time: <column if method=did, else NA>
group: <column if method=did, else NA>
mediator: <column if method=frontdoor, else NA>
</variables>
<answer>one sentence: the expected direction/sign of the effect (the number is computed by the tool, not by you)</answer>

Rules:
- Fill the confounders slot ONLY for ols/psm/glm. For iv/rdd/did/frontdoor it is NA.
- Every method-specific slot required by the chosen method MUST be a real column name; a method missing its required slot is invalid.
- Use exact column names throughout."""

CAUSCI_USER = """{description}

## Dataset columns
{columns}

## Question
{question}

Work through the tags in order: assess the study design in <cues>, choose one method from the menu in <method_reasoning>/<method>, then fill only that method's variable slots using exact column names. Do not write code."""


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
