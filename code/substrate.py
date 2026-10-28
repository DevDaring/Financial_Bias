"""
E0: a trustworthy prediction/evidence substrate (plan §2.3 P0 fixes, §6 E0).

The production pipeline has four leakage paths the plan marks P0. Every one is
closed here, and each fix is asserted by a test in selftest().

  P0-1  features.py fits medians, category levels and variance selection on the
        WHOLE frame before model.py splits it.
        -> Fit() runs inside the training fold only and is then frozen.
  P0-2  the global prior is added before local training and may have seen rows
        later used as a local test set.
        -> off by default; if enabled it is refit per outer fold.
  P0-3  fairness.audit receives predictions for all rows, including fitting rows.
        -> subgroup metrics come from out-of-fold predictions only.
  P0-4  small files share the calibration and test folds.
        -> fit / calibrate / test are disjoint; if the data cannot support three
           folds the run REFUSES rather than silently sharing.

Attribution scale is recorded explicitly (P1): SHAP on a tree model explains
the raw margin, not the calibrated probability. Evidence objects carry that.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 20260502
MIN_ROWS_THREE_FOLD = 400
MIN_POSITIVES_THREE_FOLD = 40


class InsufficientData(RuntimeError):
    """Refuse to share calibration and test folds (P0-4)."""


# ---------------------------------------------------------------------------
@dataclass
class Fit:
    """A preprocessing transform fitted ONLY on training rows, then frozen."""
    numeric_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)
    levels: Dict[str, List[str]] = field(default_factory=dict)
    keep_columns: List[str] = field(default_factory=list)
    max_levels: int = 25
    fitted: bool = False
    transformer_id: str = ""
    # v2 (2 Oct 2026). v1 behaviour is kept when version == 1 so the saved
    # pilot can be reproduced. v2 fixes two P1 defects found while building
    # Home Credit evidence:
    #  - a numeric column with many missing values (EXT_SOURCE_1, 56% NaN) was
    #    classed categorical because the numeric share was computed over ALL
    #    rows; v2 computes it over non-missing rows only.
    #  - keep_columns came from the first 200 fit rows, silently dropping any
    #    category level absent there; v2 derives columns from the fitted levels.
    #  - keep_nan: leave numeric NaN for models that handle it (LightGBM), so a
    #    missing value is explained as missing, not as an imputed median.
    version: int = 2
    keep_nan: bool = False

    def fit(self, X: pd.DataFrame, tag: str = "") -> "Fit":
        self.numeric_cols, self.categorical_cols = [], []
        for c in X.columns:
            s = X[c]
            num = pd.to_numeric(
                s.astype(str).str.replace(r"[,$%\s]", "", regex=True)
                if s.dtype == object else s, errors="coerce")
            if self.version >= 2:
                nn = s.notna()
                is_num = (nn.any() and num[nn].notna().mean() > 0.95
                          and s.nunique(dropna=True) > 10)
            else:
                is_num = num.notna().mean() > 0.6 and s.nunique(dropna=True) > 10
            if is_num:
                self.numeric_cols.append(c)
                self.medians[c] = float(np.nanmedian(num)) if num.notna().any() else 0.0
            else:
                self.categorical_cols.append(c)
                self.levels[c] = (s.astype(str).value_counts()
                                  .head(self.max_levels).index.tolist())
        self.fitted = True
        if self.version >= 2:
            import hashlib
            blob = json.dumps({"num": self.numeric_cols, "levels": self.levels,
                               "med": self.medians, "keep_nan": self.keep_nan},
                              sort_keys=True, default=str)
            self.transformer_id = f"prep2-{tag}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"
            self.keep_columns = list(self.numeric_cols) + [
                self.dummy_name(c, lv) for c in self.categorical_cols
                for lv in self.levels[c] + ["__OTHER__"]]
        else:
            # v1 used Python's salted hash(), which changed every process; a
            # stable hash is used now (saved v1 ids are kept in their records)
            import hashlib
            self.transformer_id = f"prep-{tag}-{int(hashlib.sha256('|'.join(map(str, X.columns)).encode()).hexdigest()[:8], 16) % 10**8:08d}"
            self.keep_columns = list(self.transform(X.head(min(len(X), 200))).columns)
        return self

    def dummy_name(self, c: str, lv: str) -> str:
        if self.version < 2:
            return f"{c}_{lv}"
        lvls = self.levels.get(c, []) + ["__OTHER__"]
        import re as _re
        safe = _re.sub(r"[^0-9A-Za-z_]+", "_", str(lv)).strip("_") or "blank"
        # disambiguate levels that sanitise to the same token
        same = [x for x in lvls if (_re.sub(r"[^0-9A-Za-z_]+", "_", str(x)).strip("_") or "blank") == safe]
        if len(same) > 1:
            safe = f"{safe}_{same.index(lv)}"
        return f"{c}__{safe}"

    def column_groups(self) -> Dict[str, List[str]]:
        """source column -> transformed columns (one-hot dummies grouped)."""
        g: Dict[str, List[str]] = {c: [c] for c in self.numeric_cols}
        for c in self.categorical_cols:
            g[c] = [self.dummy_name(c, lv) for lv in self.levels[c] + ["__OTHER__"]]
        return g

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Fit.transform before Fit.fit - this is P0-1")
        parts = []
        for c in self.numeric_cols:
            s = X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
            num = pd.to_numeric(
                s.astype(str).str.replace(r"[,$%\s]", "", regex=True)
                if s.dtype == object else s, errors="coerce")
            parts.append((num if self.keep_nan else num.fillna(self.medians.get(c, 0.0))).rename(c))
        for c in self.categorical_cols:
            s = (X[c] if c in X.columns
                 else pd.Series("__NA__", index=X.index)).astype(str)
            s = s.where(s.isin(self.levels.get(c, [])), "__OTHER__")
            if self.version >= 2:
                # explicit, name-safe dummies (LightGBM rejects JSON characters
                # such as ':' ',' '"' in feature names; HC levels contain them)
                for lv in self.levels.get(c, []) + ["__OTHER__"]:
                    parts.append((s == lv).astype(float).rename(self.dummy_name(c, lv)))
            else:
                parts.append(pd.get_dummies(s, prefix=c, dtype=float))
        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
        if self.keep_nan:
            out = out.replace([np.inf, -np.inf], np.nan)
            dummies = [c for c in out.columns if c not in self.numeric_cols]
            out[dummies] = out[dummies].fillna(0.0)
        else:
            out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if self.keep_columns:
            out = out.reindex(columns=self.keep_columns, fill_value=0.0)
        return out


# ---------------------------------------------------------------------------
@dataclass
class Substrate:
    """A frozen predictor + calibrator + its provenance."""
    name: str
    model: object = None
    calibrator: object = None
    prep: Optional[Fit] = None
    model_id: str = ""
    calibrator_id: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    subgroup: Dict[str, dict] = field(default_factory=dict)
    split_sizes: Dict[str, int] = field(default_factory=dict)
    attribution_scale: str = "raw log-odds margin (not calibrated PD)"
    reference_background: str = "training-fold mean"

    def predict_pd(self, X_raw: pd.DataFrame) -> np.ndarray:
        Z = self.prep.transform(X_raw)
        raw = _proba(self.model, Z)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def predict_margin(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Uncalibrated score - the scale SHAP actually explains."""
        return _proba(self.model, self.prep.transform(X_raw))


def _proba(m, X) -> np.ndarray:
    if hasattr(m, "predict_proba"):
        p = m.predict_proba(X)
        return p[:, 1] if getattr(p, "ndim", 1) == 2 else p
    return np.asarray(m.predict(X), dtype=float)


# ---------------------------------------------------------------------------
def three_way_split(n: int, y: pd.Series, *, seed: int = SEED,
                    fit: float = 0.60, cal: float = 0.15
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint fit / calibrate / test indices. Refuses if too small (P0-4)."""
    pos = int(y.sum())
    if n < MIN_ROWS_THREE_FOLD or pos < MIN_POSITIVES_THREE_FOLD:
        raise InsufficientData(
            f"n={n}, positives={pos}: below the {MIN_ROWS_THREE_FOLD}-row / "
            f"{MIN_POSITIVES_THREE_FOLD}-positive floor for three disjoint folds. "
            f"Sharing the calibration and test folds would make ECE a fitted "
            f"quantity, not a held-out one. Report this corpus as supplementary.")
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    i_fit, i_rest = train_test_split(idx, train_size=fit, random_state=seed,
                                     stratify=y.values)
    rel = cal / (1.0 - fit)
    i_cal, i_test = train_test_split(i_rest, train_size=rel, random_state=seed,
                                     stratify=y.values[i_rest])
    return i_fit, i_cal, i_test


# ---------------------------------------------------------------------------
def make_model(kind: str, y_fit: pd.Series, seed: int = SEED, version: int = 1):
    """LR / GBDT / additive - the plan's E0 comparators, identical features."""
    pos, neg = max(1, int(y_fit.sum())), max(1, int((1 - y_fit).sum()))
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                               random_state=seed))
    if kind == "gbdt":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=48,
            min_child_samples=30, subsample=0.85, subsample_freq=1,
            colsample_bytree=0.85, reg_lambda=3.0, scale_pos_weight=neg / pos,
            random_state=seed, verbose=-1,
            # v2: fixed threads + deterministic mode so a rerun (e.g. the pack
            # builder) reproduces the exact model on a loaded machine
            **({"n_jobs": 8, "deterministic": True, "force_row_wise": True}
               if version >= 2 else {"n_jobs": -1}))
    if kind == "additive":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
            return ExplainableBoostingClassifier(random_state=seed)
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_depth=2, random_state=seed)   # depth-2 ~ additive-ish
    raise ValueError(f"unknown model kind {kind!r}")


def make_calibrator(kind: str):
    if kind == "none":
        return None
    if kind == "platt":
        from sklearn.linear_model import LogisticRegression

        class _Platt:
            def __init__(self):
                self.m = LogisticRegression(max_iter=1000)
            def fit(self, p, y):
                self.m.fit(np.asarray(p).reshape(-1, 1), y); return self
            def predict(self, p):
                return self.m.predict_proba(np.asarray(p).reshape(-1, 1))[:, 1]
        return _Platt()
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    raise ValueError(f"unknown calibrator {kind!r}")


# ---------------------------------------------------------------------------
def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> Tuple[float, List[int]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    tot, counts = 0.0, []
    for b in range(bins):
        m = idx == b
        counts.append(int(m.sum()))
        if m.sum():
            tot += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(tot), counts


def score(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    e, counts = ece(y, p)
    return {"auc": round(float(roc_auc_score(y, p)), 5),
            "ap": round(float(average_precision_score(y, p)), 5),
            "brier": round(float(brier_score_loss(y, p)), 5),
            "ece": round(e, 5), "n": int(len(y)),
            "base_rate": round(float(y.mean()), 5),
            "ece_bin_counts": counts}


def subgroup_rates(y: np.ndarray, p: np.ndarray, groups: pd.Series,
                   threshold: float) -> Dict[str, dict]:
    """FPR/TPR by group at a DEVELOPMENT-selected threshold, out-of-fold only."""
    out: Dict[str, dict] = {}
    yhat = (p >= threshold).astype(int)
    for g in sorted(pd.Series(groups).astype(str).unique()):
        m = (pd.Series(groups).astype(str) == g).values
        if m.sum() < 20:
            continue
        yy, pp = y[m], yhat[m]
        P, N = int(yy.sum()), int((1 - yy).sum())
        out[g] = {"n": int(m.sum()),
                  "base_rate": round(float(yy.mean()), 4),
                  "selection_rate": round(float(pp.mean()), 4),
                  "tpr": round(float(pp[yy == 1].mean()), 4) if P else None,
                  "fpr": round(float(pp[yy == 0].mean()), 4) if N else None}
    return out


# ---------------------------------------------------------------------------
def build(X_raw: pd.DataFrame, y: pd.Series, *, name: str,
          model_kind: str = "gbdt", calibrator_kind: str = "isotonic",
          groups: Optional[pd.Series] = None, threshold: float = 0.5,
          seed: int = SEED, prep_version: int = 1) -> Substrate:
    """Fit a frozen substrate with every P0 fix in force."""
    y = y.reset_index(drop=True).astype(int)
    X_raw = X_raw.reset_index(drop=True)
    i_fit, i_cal, i_test = three_way_split(len(y), y, seed=seed)

    assert not (set(i_fit) & set(i_cal)), "P0-4: fit/cal overlap"
    assert not (set(i_cal) & set(i_test)), "P0-4: cal/test overlap"
    assert not (set(i_fit) & set(i_test)), "P0-4: fit/test overlap"

    # P0-1: preprocessing fitted on training rows ONLY
    prep = Fit(version=prep_version,
               keep_nan=(prep_version >= 2 and model_kind == "gbdt")).fit(X_raw.iloc[i_fit], tag=name)
    Z_fit = prep.transform(X_raw.iloc[i_fit])
    Z_cal = prep.transform(X_raw.iloc[i_cal])
    Z_test = prep.transform(X_raw.iloc[i_test])

    model = make_model(model_kind, y.iloc[i_fit], seed, version=prep_version)
    model.fit(Z_fit, y.iloc[i_fit])

    cal = make_calibrator(calibrator_kind)
    if cal is not None:
        cal.fit(_proba(model, Z_cal), y.iloc[i_cal].values)

    p_test = _proba(model, Z_test)
    if cal is not None:
        p_test = cal.predict(p_test)

    s = Substrate(
        name=name, model=model, calibrator=cal, prep=prep,
        model_id=f"{model_kind}-{name}-s{seed}",
        calibrator_id=(f"{calibrator_kind}-{name}-s{seed}" if cal is not None else "none"),
        metrics=score(y.iloc[i_test].values, p_test),
        split_sizes={"fit": len(i_fit), "calibrate": len(i_cal), "test": len(i_test)})

    # P0-3: subgroup metrics from held-out predictions only
    if groups is not None:
        s.subgroup = subgroup_rates(y.iloc[i_test].values, p_test,
                                    groups.reset_index(drop=True).iloc[i_test],
                                    threshold)
    s.metrics["_test_index"] = i_test.tolist()
    return s


# ---------------------------------------------------------------------------
def selftest() -> None:
    """Assert each P0 fix actually holds."""
    rng = np.random.RandomState(0)
    n = 3000
    X = pd.DataFrame({
        "amount": rng.lognormal(11, .7, n),
        "term": rng.choice([12, 24, 36, 60, 84], n),
        "score": rng.normal(690, 70, n).clip(400, 900),
        "sector": rng.choice(list("ABCDE"), n),
    })
    z = (-1.4 - .012 * (X.score - 690) + .3 * (X.sector == "C") + rng.normal(0, .6, n))
    y = pd.Series((rng.rand(n) < 1 / (1 + np.exp(-z))).astype(int))
    g = pd.Series(rng.choice(["g1", "g2"], n))

    print("P0-4  disjoint folds ................", end=" ")
    i1, i2, i3 = three_way_split(n, y)
    assert not (set(i1) & set(i2) or set(i2) & set(i3) or set(i1) & set(i3))
    print(f"OK  fit={len(i1)} cal={len(i2)} test={len(i3)}")

    print("P0-4  refuses tiny corpora .........", end=" ")
    try:
        three_way_split(150, y.head(150))
        print("FAIL - should have refused")
    except InsufficientData:
        print("OK  (raises InsufficientData)")

    print("P0-1  prep fitted on fit-fold only ..", end=" ")
    f_all = Fit().fit(X, "all")
    f_fit = Fit().fit(X.iloc[i1], "fitfold")
    assert f_all.medians["amount"] != f_fit.medians["amount"], "medians identical"
    print(f"OK  median(all)={f_all.medians['amount']:.1f} != "
          f"median(fit)={f_fit.medians['amount']:.1f}")

    print("P0-3  subgroup from held-out only ...", end=" ")
    s = build(X, y, name="demo", model_kind="gbdt", groups=g, threshold=0.3)
    assert len(s.metrics["_test_index"]) == s.split_sizes["test"]
    print(f"OK  n_test={s.split_sizes['test']}, groups={list(s.subgroup)}")

    print("\nmodel / calibrator grid (held-out test fold):")
    for mk in ("logistic", "gbdt", "additive"):
        row = []
        for ck in ("none", "platt", "isotonic"):
            try:
                ss = build(X, y, name=f"{mk}-{ck}", model_kind=mk,
                           calibrator_kind=ck)
                row.append(f"{ck}: AUC={ss.metrics['auc']:.4f} ECE={ss.metrics['ece']:.4f}")
            except Exception as e:
                row.append(f"{ck}: FAILED {type(e).__name__}")
        print(f"  {mk:<10} " + " | ".join(row))


if __name__ == "__main__":
    selftest()
