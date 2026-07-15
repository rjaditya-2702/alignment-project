"""verify.py — numeric CBN expression-equivalence verifier (rung-agnostic; replaces DoVerifier).

The Step-3 identification check: is the model's do-free derived expression FUNCTIONALLY equivalent
to the gold estimable form (reasoning.step3)? We build N random binary parameterizations (CBNs) of
the item's graph structure, evaluate BOTH expressions observationally on each, and require exact
agreement on all. Two expressions that correctly identify the same quantity are equal functions of
the observational distribution → agree on every parameterization; an expression that's merely
numerically lucky on one CBN breaks on a re-draw. Works identically for rungs 1/2/3 (the gold
estimable form is do-free and counterfactual-free), so it replaces the symbolic do-calculus verifier.

Also keeps light string helpers (`adjustment_set`) used by the CauSci transfer eval.
"""

import itertools
import random
import re

import networkx as nx

_VAR = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_PGROUP = re.compile(r"[EP]\s*[\[(]([^\])]*)[\])]")
_SUM = re.compile(r"sum_\{\s*(\w+)\s*=\s*(\w+)\s*\}")
_PATOM = re.compile(r"[EP]\s*\(([^()]*)\)")


# ── graph → CBN → joint ─────────────────────────────────────────────────────

def _parse_graph(edge_str):
    """'V1->X,X->Y' -> (nodes in topo order, {node: [parents]}). () if not a DAG."""
    g = nx.DiGraph()
    for e in str(edge_str).replace("→", "->").split(","):
        if "->" in e:
            a, b = (t.strip() for t in e.split("->", 1))
            if a and b:
                g.add_edge(a, b)
    if g.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(g):
        return [], {}
    nodes = list(nx.topological_sort(g))
    return nodes, {n: list(g.predecessors(n)) for n in nodes}


def _random_cbn(nodes, parents, seed):
    """Random CPT per node: P(node=1 | parents) ~ U(0.1, 0.9) for each parent assignment."""
    rng = random.Random(seed)
    cpt = {}
    for n in nodes:
        cpt[n] = {combo: rng.uniform(0.1, 0.9)
                  for combo in itertools.product((0, 1), repeat=len(parents[n]))}
    return cpt


def _joint(nodes, parents, cpt):
    """Full joint as {assignment_tuple (node order): prob} over all 2^|nodes| binary assignments."""
    idx = {n: i for i, n in enumerate(nodes)}
    joint = {}
    for a in itertools.product((0, 1), repeat=len(nodes)):
        p = 1.0
        for n in nodes:
            pv = cpt[n][tuple(a[idx[pa]] for pa in parents[n])]
            p *= pv if a[idx[n]] == 1 else (1 - pv)
        joint[a] = p
    return joint, idx


def _P(joint, idx, event):
    """P(event) where event = {var: val}. Raises KeyError if a var isn't in the graph."""
    keys = [(idx[v], val) for v, val in event.items()]
    return sum(p for a, p in joint.items() if all(a[i] == val for i, val in keys))


# ── expression evaluation ────────────────────────────────────────────────────

def _subst(text, var, dummy, val):
    return re.sub(rf"{re.escape(var)}\s*=\s*{re.escape(dummy)}(?![0-9A-Za-z])", f"{var}={val}", text)


def _assign(part):
    """'Y=1,V1=0' -> {'Y':1,'V1':0}; bare 'Y' -> {'Y':1} (E[]-style)."""
    d = {}
    for a in part.split(","):
        a = a.strip()
        if not a:
            continue
        if "=" in a:
            k, v = a.split("=", 1)
            d[k.strip()] = int(v)                 # ValueError propagates (unexpanded dummy) → not equiv
        elif _VAR.fullmatch(a):
            d[a] = 1
    return d


def eval_expr(expr, joint, idx):
    """Evaluate a do-free probability expression on a joint distribution. Σ_{V=v} quantifiers are
    handled by iterating the Cartesian product of dummy values over ONE body (2^k evals; no string
    duplication → no blowup on nested sums like the frontdoor formula)."""
    s = str(expr).replace("−", "-").replace("Σ", "sum").replace("\\sum", "sum").replace("\\", "")
    quants = _SUM.findall(s)                                 # [(var, dummy), ...]
    body = _SUM.sub("", s).replace("[", "(").replace("]", ")")

    def p(inner):
        inner = inner.replace(" ", "")
        out, cond = (inner.split("|", 1) + [""])[:2]
        ev, cd = _assign(out), _assign(cond)
        den = _P(joint, idx, cd) if cd else 1.0
        return _P(joint, idx, {**cd, **ev}) / den if den > 0 else float("nan")

    total = 0.0
    for combo in itertools.product((0, 1), repeat=len(quants)):
        b = body
        for (var, dummy), val in zip(quants, combo):
            b = _subst(b, var, dummy, val)
        b = _PATOM.sub(lambda m: f'p("{m.group(1)}")', b)
        total += eval(b, {"__builtins__": {}}, {"p": p})
    return total


# ── public API ───────────────────────────────────────────────────────────────

def is_estimable(expr):
    """True if the expression is reduced to an estimable form: no do(), and no counterfactual
    subscript (Y_{X=1}). Sum quantifiers (Σ_{V=v}) are fine, so strip them before the subscript
    check. The Step-3 do-free gate."""
    e = str(expr)
    if "do(" in e:
        return False
    e = re.sub(r"(\\?sum|Σ)\s*_\{[^}]*\}", "", e)   # drop sum quantifiers
    return "_{" not in e                            # any remaining _{ is a counterfactual subscript


def evaluable(expr, edge_str, seed=0):
    """Does this expression evaluate without error on one random CBN of the graph? (used to keep
    only records whose GOLD estimable form is numerically checkable)."""
    nodes, parents = _parse_graph(edge_str)
    if not nodes:
        return False
    joint, idx = _joint(nodes, parents, _random_cbn(nodes, parents, seed))
    try:
        v = eval_expr(expr, joint, idx)
        return v == v            # not NaN
    except Exception:
        return False


def expr_equiv(model_expr, gold_expr, edge_str, n=12, tol=1e-6):
    """True iff model_expr ≡ gold_expr as functions of the observational distribution — agreement
    on all n random parameterizations of the graph. Both must be do-free/estimable."""
    nodes, parents = _parse_graph(edge_str)
    if not nodes or not is_estimable(model_expr):
        return False
    for i in range(n):
        joint, idx = _joint(nodes, parents, _random_cbn(nodes, parents, seed=1000 + i))
        try:
            mv, gv = eval_expr(model_expr, joint, idx), eval_expr(gold_expr, joint, idx)
        except Exception:
            return False
        if mv != mv or gv != gv or abs(mv - gv) > tol:   # NaN or disagreement
            return False
    return True


# ── string helper for the CauSci transfer eval ──────────────────────────────

def adjustment_set(identify_str, treatment, outcome):
    """(treated, Z): did any term condition the outcome on the treatment, and the adjustment set.
    'P(Y|X)' -> (True, {}); 'Σ_{V1} P(V1)[P(Y|V1,X=1)-...]' -> (True, {V1})."""
    treated, adj = False, set()
    for inner in _PGROUP.findall(str(identify_str)):
        if "|" not in inner:
            continue
        cond = re.sub(r"do\(([^)]*)\)", r"\1", inner.split("|", 1)[1])
        names = {v.split("=")[0].strip() for v in cond.split(",")}
        names = {c for c in names if _VAR.fullmatch(c or "")}
        if treatment in names:
            treated = True
            adj |= names
    adj.discard(treatment)
    adj.discard(outcome)
    return treated, adj
