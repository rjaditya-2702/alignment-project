"""
Reward extraction tests for all three training scripts.

Stubs heavy ML / scientific deps at sys.modules level so training modules
load without GPU, model weights, or a running judge server.

Coverage:
  - extract_cladder   : TRL, veRL
  - extract_causci    : TRL, SFT, veRL
  - reward_fn         : TRL (batch_judge mocked), veRL (batch_judge mocked)
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── stub heavy deps before any training module is imported ───────────────────
_STUBS = [
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.parallel",
    "torch.utils", "torch.utils.data", "torch.utils.data.distributed",
    "torch.distributed", "torch.optim",
    "torch.amp", "torch.cuda", "torch.cuda.amp",
    "trl", "peft",
    "transformers",
    "datasets",
    "numpy", "numpy.random",
    "pandas",
    "matplotlib", "matplotlib.pyplot",
    "openai",
    "statsmodels", "statsmodels.formula", "statsmodels.formula.api", "statsmodels.api",
    "sklearn", "sklearn.linear_model", "sklearn.preprocessing",
    "linearmodels", "linearmodels.iv", "linearmodels.panel",
    "scipy", "scipy.stats",
    "src.training.tool_calling",
]
for _m in _STUBS:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

# torch.bfloat16 is used as a dtype constant — needs a stable identity
sys.modules["torch"].bfloat16 = object()

# ── import training modules ───────────────────────────────────────────────────
import importlib
trl_mod  = importlib.import_module("src.training.train_trl")
sft_mod  = importlib.import_module("src.training.train_sft_ddp")
verl_mod = importlib.import_module("src.training.verl_.reward")


# ── shared test data ──────────────────────────────────────────────────────────

CAUSCI_COLS = ["education", "salary", "age"]

VALID_CLADDER = {
    "step1": "Let X = education; Y = salary. X->Y",
    "step2": "ate",
    "step3": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    "step4": "0.79 - 0.43 = 0.36 > 0",
    "step5": "yes",
}

VALID_CAUSCI = {
    "step1": {
        "treatment": "education",
        "outcome":   "salary",
        "controls":  ["age"],
        "instrument": None, "running_variable": None, "cutoff": None,
        "time_variable": None, "group_variable": None, "mediator": None,
        "estimand": "ate",
    },
    "step2": "ols",
}

def _raw(d):           return json.dumps(d)
def _think(d):         return "<think>reasoning here</think>" + json.dumps(d)
def _trailing(d):      return json.dumps(d).replace("}", ",}")   # trailing comma


# ══════════════════════════════════════════════════════════════════════════════
# extract_cladder — TRL and veRL
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mod,label", [
    (trl_mod,  "TRL"),
    (verl_mod, "veRL"),
])
class TestExtractCladder:

    def test_valid_clean_json(self, mod, label):
        result = mod.extract_cladder(_raw(VALID_CLADDER))
        assert result is not None, f"{label}: expected parsed dict"
        assert result["step2"] == "ate"
        assert result["step5"] == "yes"

    def test_valid_after_think_block(self, mod, label):
        result = mod.extract_cladder(_think(VALID_CLADDER))
        assert result is not None, f"{label}: should strip </think> block"
        assert result["step2"] == "ate"

    def test_trailing_comma_cleaned(self, mod, label):
        raw = _trailing(VALID_CLADDER)
        result = mod.extract_cladder(raw)
        assert result is not None, f"{label}: should recover trailing comma"

    def test_missing_required_field(self, mod, label):
        d = {k: v for k, v in VALID_CLADDER.items() if k != "step3"}
        assert mod.extract_cladder(_raw(d)) is None

    def test_unknown_query_type(self, mod, label):
        d = {**VALID_CLADDER, "step2": "anova"}
        assert mod.extract_cladder(_raw(d)) is None

    def test_invalid_step5(self, mod, label):
        d = {**VALID_CLADDER, "step5": "maybe"}
        assert mod.extract_cladder(_raw(d)) is None

    def test_step2_case_insensitive(self, mod, label):
        d = {**VALID_CLADDER, "step2": "ATE"}
        result = mod.extract_cladder(_raw(d))
        assert result is not None
        assert result["step2"] == "ate"

    def test_step5_case_insensitive(self, mod, label):
        d = {**VALID_CLADDER, "step5": "YES"}
        result = mod.extract_cladder(_raw(d))
        assert result is not None
        assert result["step5"] == "yes"

    def test_no_json_in_output(self, mod, label):
        assert mod.extract_cladder("no json here at all") is None

    def test_leading_text(self, mod, label):
        output = "Sure, here is my answer:\n" + _raw(VALID_CLADDER)
        result = mod.extract_cladder(output)
        assert result is not None, f"{label}: should parse JSON with leading text"
        assert result["step2"] == "ate"

    def test_trailing_text(self, mod, label):
        output = _raw(VALID_CLADDER) + "\nI hope this helps!"
        result = mod.extract_cladder(output)
        assert result is not None, f"{label}: should parse JSON with trailing text"
        assert result["step2"] == "ate"

    def test_leading_and_trailing_text(self, mod, label):
        output = "Let me solve this step by step.\n" + _raw(VALID_CLADDER) + "\nLet me know if you need more."
        result = mod.extract_cladder(output)
        assert result is not None, f"{label}: should parse JSON with leading and trailing text"
        assert result["step5"] == "yes"

    def test_all_query_types_accepted(self, mod, label):
        for qt in ["marginal", "correlation", "ate", "backadj",
                   "det-counterfactual", "ett", "nde", "nie",
                   "collider_bias", "exp_away"]:
            d = {**VALID_CLADDER, "step2": qt}
            assert mod.extract_cladder(_raw(d)) is not None, f"{label}: {qt} rejected"


# ══════════════════════════════════════════════════════════════════════════════
# extract_causci — TRL, SFT, veRL
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("mod,label", [
    (trl_mod,  "TRL"),
    (sft_mod,  "SFT"),
    (verl_mod, "veRL"),
])
class TestExtractCausci:

    def test_valid_ols(self, mod, label):
        result = mod.extract_causci(_raw(VALID_CAUSCI), CAUSCI_COLS)
        assert result is not None, f"{label}: valid OLS should parse"
        assert result["step2"] == "ols"
        assert result["step1"]["treatment"] == "education"
        assert result["step1"]["outcome"]   == "salary"
        assert result["step1"]["controls"]  == ["age"]

    def test_valid_after_think_block(self, mod, label):
        result = mod.extract_causci(_think(VALID_CAUSCI), CAUSCI_COLS)
        assert result is not None, f"{label}: should strip </think>"

    def test_unknown_method(self, mod, label):
        d = {**VALID_CAUSCI, "step2": "lasso"}
        assert mod.extract_causci(_raw(d), CAUSCI_COLS) is None

    def test_treatment_not_in_columns(self, mod, label):
        d = {**VALID_CAUSCI,
             "step1": {**VALID_CAUSCI["step1"], "treatment": "unknown_col"}}
        assert mod.extract_causci(_raw(d), CAUSCI_COLS) is None

    def test_outcome_not_in_columns(self, mod, label):
        d = {**VALID_CAUSCI,
             "step1": {**VALID_CAUSCI["step1"], "outcome": "unknown_col"}}
        assert mod.extract_causci(_raw(d), CAUSCI_COLS) is None

    def test_controls_filtered_to_valid(self, mod, label):
        d = {**VALID_CAUSCI,
             "step1": {**VALID_CAUSCI["step1"], "controls": ["age", "ghost_col"]}}
        result = mod.extract_causci(_raw(d), CAUSCI_COLS)
        assert result is not None
        assert result["step1"]["controls"] == ["age"]   # ghost_col stripped

    def test_no_json(self, mod, label):
        assert mod.extract_causci("no json", CAUSCI_COLS) is None

    def test_leading_text(self, mod, label):
        output = "Based on the study design:\n" + _raw(VALID_CAUSCI)
        result = mod.extract_causci(output, CAUSCI_COLS)
        assert result is not None, f"{label}: should parse JSON with leading text"
        assert result["step2"] == "ols"

    def test_trailing_text(self, mod, label):
        output = _raw(VALID_CAUSCI) + "\nPlease let me know if this is correct."
        result = mod.extract_causci(output, CAUSCI_COLS)
        assert result is not None, f"{label}: should parse JSON with trailing text"
        assert result["step2"] == "ols"

    def test_leading_and_trailing_text(self, mod, label):
        output = "My analysis:\n" + _raw(VALID_CAUSCI) + "\nEnd of response."
        result = mod.extract_causci(output, CAUSCI_COLS)
        assert result is not None, f"{label}: should parse JSON with leading and trailing text"
        assert result["step1"]["treatment"] == "education"

    def test_iv_invalid_instrument(self, mod, label):
        d = {
            "step1": {**VALID_CAUSCI["step1"], "instrument": "missing_instrument"},
            "step2": "iv",
        }
        assert mod.extract_causci(_raw(d), CAUSCI_COLS) is None

    def test_rdd_missing_cutoff(self, mod, label):
        cols = CAUSCI_COLS + ["score"]
        d = {
            "step1": {**VALID_CAUSCI["step1"],
                      "running_variable": "score", "cutoff": None},
            "step2": "rdd",
        }
        assert mod.extract_causci(_raw(d), cols) is None

    def test_rdd_valid(self, mod, label):
        cols = CAUSCI_COLS + ["score"]
        d = {
            "step1": {**VALID_CAUSCI["step1"],
                      "running_variable": "score", "cutoff": 50.0},
            "step2": "rdd",
        }
        result = mod.extract_causci(_raw(d), cols)
        assert result is not None
        assert result["step2"] == "rdd"

    def test_did_valid(self, mod, label):
        cols = CAUSCI_COLS + ["year", "group"]
        d = {
            "step1": {**VALID_CAUSCI["step1"],
                      "time_variable": "year", "group_variable": "group"},
            "step2": "did",
        }
        result = mod.extract_causci(_raw(d), cols)
        assert result is not None
        assert result["step2"] == "did"

    def test_frontdoor_valid(self, mod, label):
        cols = CAUSCI_COLS + ["mediator_col"]
        d = {
            "step1": {**VALID_CAUSCI["step1"], "mediator": "mediator_col"},
            "step2": "frontdoor",
        }
        result = mod.extract_causci(_raw(d), cols)
        assert result is not None
        assert result["step2"] == "frontdoor"


# ══════════════════════════════════════════════════════════════════════════════
# reward_fn — TRL
# ══════════════════════════════════════════════════════════════════════════════

def _trl_cladder_call(completion, gt_dict, judge_scores):
    gt_str = json.dumps(gt_dict)
    with patch.object(trl_mod, "batch_judge", return_value=judge_scores):
        return trl_mod.reward_fn(
            [completion],
            source=["cladder"],
            groundtruth=[gt_str],
            dataset_columns=[[]],
            csv_path=[""],
        )

class TestTRLRewardFn:

    def test_cladder_perfect_score(self):
        # step1 correct, step3 correct, answer matches → 1.0
        gt = {**VALID_CLADDER}
        rewards = _trl_cladder_call(_raw(VALID_CLADDER), gt, [1.0, 1.0])
        assert rewards[0] == pytest.approx(1.0)

    def test_cladder_correct_answer_wrong_estimand(self):
        # step1 correct, step3 wrong → 1.0 - 0.25 = 0.75
        gt = {**VALID_CLADDER}
        rewards = _trl_cladder_call(_raw(VALID_CLADDER), gt, [1.0, 0.0])
        assert rewards[0] == pytest.approx(0.75)

    def test_cladder_wrong_answer(self):
        # step1 correct, step3 correct, answer wrong → -0.75
        d = {**VALID_CLADDER, "step5": "no"}
        gt = {**VALID_CLADDER}       # gt says "yes"
        rewards = _trl_cladder_call(_raw(d), gt, [1.0, 1.0])
        assert rewards[0] == pytest.approx(-0.75)

    def test_cladder_wrong_step1(self):
        # step1 wrong → early exit -1.0
        # step3 judge IS queued because step2 matches, but its result is ignored
        gt = {**VALID_CLADDER}
        rewards = _trl_cladder_call(_raw(VALID_CLADDER), gt, [0.0, 0.0])
        assert rewards[0] == pytest.approx(-1.0)

    def test_cladder_wrong_step2(self):
        # step1 correct, step2 misclassified → -0.5
        d = {**VALID_CLADDER, "step2": "ett"}
        gt = {**VALID_CLADDER}   # gt says "ate"
        rewards = _trl_cladder_call(_raw(d), gt, [1.0])  # only step1 judge fires
        assert rewards[0] == pytest.approx(-0.5)

    def test_cladder_unparseable_output(self):
        rewards = _trl_cladder_call("not json at all", VALID_CLADDER, [])
        assert rewards[0] == pytest.approx(-1.0)

    def test_cladder_think_block_stripped(self):
        gt = {**VALID_CLADDER}
        rewards = _trl_cladder_call(_think(VALID_CLADDER), gt, [1.0, 1.0])
        assert rewards[0] == pytest.approx(1.0)

    def test_causci_correct(self):
        gt = {
            "step1": {"treatment": "education", "outcome": "salary",
                      "controls": ["age"]},
            "step2": "ols",
            "step5": 0.5,
        }
        completion = _raw(VALID_CAUSCI)
        with patch.object(trl_mod, "cached_library_fn", return_value=0.5):
            rewards = trl_mod.reward_fn(
                [completion],
                source=["causcibench"],
                groundtruth=[json.dumps(gt)],
                dataset_columns=[CAUSCI_COLS],
                csv_path=["some/path.csv"],
            )
        assert rewards[0] == pytest.approx(1.0)

    def test_causci_wrong_method(self):
        gt = {"step1": {"treatment": "education", "outcome": "salary",
                        "controls": ["age"]},
              "step2": "ipw", "step5": 0.5}
        completion = _raw(VALID_CAUSCI)   # predicts "ols", gt wants "ipw"
        with patch.object(trl_mod, "cached_library_fn", return_value=0.5):
            rewards = trl_mod.reward_fn(
                [completion],
                source=["causcibench"],
                groundtruth=[json.dumps(gt)],
                dataset_columns=[CAUSCI_COLS],
                csv_path=["some/path.csv"],
            )
        assert rewards[0] == pytest.approx(-1.0)

    def test_mixed_batch(self):
        cladder_gt = json.dumps(VALID_CLADDER)
        causci_gt  = json.dumps({"step1": {"treatment": "education",
                                           "outcome": "salary",
                                           "controls": ["age"]},
                                 "step2": "ols", "step5": 0.5})
        with patch.object(trl_mod, "batch_judge", return_value=[1.0, 1.0]), \
             patch.object(trl_mod, "cached_library_fn", return_value=0.5):
            rewards = trl_mod.reward_fn(
                [_raw(VALID_CLADDER), _raw(VALID_CAUSCI)],
                source=["cladder", "causcibench"],
                groundtruth=[cladder_gt, causci_gt],
                dataset_columns=[[], CAUSCI_COLS],
                csv_path=["", "some/path.csv"],
            )
        assert len(rewards) == 2
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# reward_fn — veRL
# ══════════════════════════════════════════════════════════════════════════════

def _verl_extra(csv_path="", cols=None):
    return json.dumps({"csv_path": csv_path, "dataset_columns": cols or []})

def _verl_gt(d):
    return json.dumps({"ground_truth": d})

def _verl_cladder_call(completion, gt_dict, judge_scores):
    with patch.object(verl_mod, "batch_judge", return_value=judge_scores):
        return verl_mod.reward_fn(
            [completion],
            [_verl_gt(gt_dict)],
            [_verl_extra()],
        )

class TestVeRLRewardFn:

    def test_cladder_perfect_score(self):
        rewards = _verl_cladder_call(_raw(VALID_CLADDER), VALID_CLADDER, [1.0, 1.0])
        assert rewards[0] == pytest.approx(1.0)

    def test_cladder_correct_answer_wrong_estimand(self):
        rewards = _verl_cladder_call(_raw(VALID_CLADDER), VALID_CLADDER, [1.0, 0.0])
        assert rewards[0] == pytest.approx(0.75)

    def test_cladder_wrong_answer(self):
        d = {**VALID_CLADDER, "step5": "no"}
        rewards = _verl_cladder_call(_raw(d), VALID_CLADDER, [1.0, 1.0])
        assert rewards[0] == pytest.approx(-0.75)

    def test_cladder_wrong_step1(self):
        # step3 judge IS queued because step2 matches, but its result is ignored
        rewards = _verl_cladder_call(_raw(VALID_CLADDER), VALID_CLADDER, [0.0, 0.0])
        assert rewards[0] == pytest.approx(-1.0)

    def test_cladder_wrong_step2(self):
        d = {**VALID_CLADDER, "step2": "ett"}
        rewards = _verl_cladder_call(_raw(d), VALID_CLADDER, [1.0])
        assert rewards[0] == pytest.approx(-0.5)

    def test_cladder_unparseable_output(self):
        rewards = _verl_cladder_call("gibberish", VALID_CLADDER, [])
        assert rewards[0] == pytest.approx(-1.0)

    def test_cladder_think_block_stripped(self):
        rewards = _verl_cladder_call(_think(VALID_CLADDER), VALID_CLADDER, [1.0, 1.0])
        assert rewards[0] == pytest.approx(1.0)

    def test_causci_correct(self):
        gt = {"step1": {"treatment": "education", "outcome": "salary",
                        "controls": ["age"]},
              "step2": "ols", "step5": 0.5}
        with patch.object(verl_mod, "cached_library_fn", return_value=0.5):
            rewards = verl_mod.reward_fn(
                [_raw(VALID_CAUSCI)],
                [_verl_gt(gt)],
                [_verl_extra("some/path.csv", CAUSCI_COLS)],
            )
        assert rewards[0] == pytest.approx(1.0)

    def test_causci_wrong_method(self):
        gt = {"step1": {"treatment": "education", "outcome": "salary",
                        "controls": ["age"]},
              "step2": "ipw", "step5": 0.5}
        with patch.object(verl_mod, "cached_library_fn", return_value=0.5):
            rewards = verl_mod.reward_fn(
                [_raw(VALID_CAUSCI)],
                [_verl_gt(gt)],
                [_verl_extra("some/path.csv", CAUSCI_COLS)],
            )
        assert rewards[0] == pytest.approx(-1.0)

    def test_mixed_batch(self):
        causci_gt = {"step1": {"treatment": "education", "outcome": "salary",
                               "controls": ["age"]},
                     "step2": "ols", "step5": 0.5}
        with patch.object(verl_mod, "batch_judge", return_value=[1.0, 1.0]), \
             patch.object(verl_mod, "cached_library_fn", return_value=0.5):
            rewards = verl_mod.reward_fn(
                [_raw(VALID_CLADDER), _raw(VALID_CAUSCI)],
                [_verl_gt(VALID_CLADDER), _verl_gt(causci_gt)],
                [_verl_extra(), _verl_extra("some/path.csv", CAUSCI_COLS)],
            )
        assert len(rewards) == 2
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[1] == pytest.approx(1.0)
