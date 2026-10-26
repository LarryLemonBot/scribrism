"""Research-only binary score candidates; never used by school scan decisions.

Requires an independently frozen study protocol and grouped out-of-fold evaluation.
Outputs are dataset-specific scores, not authenticated authorship probabilities.
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


KINDS = ('R1', 'R2')
CFGS = (256, 512, 768)
EPS = 1e-6


def features(scores, kind):
    """scores: n x 3 contexts x 4 training classes, ordered human/ai/edited/humanized."""
    if kind not in KINDS:
        raise ValueError('unknown research candidate')
    p = np.asarray(scores, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (3, 4) or not len(p):
        raise ValueError('nonempty n by 3 by 4 scores required')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('finite probability components in [0,1] required')
    if (np.abs(p.sum(axis=2) - 1) > 1e-5 + 1e-12).any():
        raise ValueError('probability sum outside remote rounding contract')
    if kind == 'R1':
        q = np.clip(1-p[:, 1, 0], EPS, 1-EPS)
        return (np.log(q)-np.log1p(-q))[:, None]
    floored = np.maximum(p, EPS)
    return np.log(floored[:, :, 1:]/floored[:, :, :1]).reshape(len(p), 9)


def _training_arrays(x, labels):
    x, y = np.asarray(x, dtype=float), np.asarray(labels)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('nonempty finite feature matrix required')
    if y.ndim != 1 or len(y) != len(x) or y.dtype.kind not in 'iu' or set(y.tolist()) != {0, 1}:
        raise ValueError('both binary integer classes required')
    return x, y


def grouped_folds(labels, groups, n_splits=5, seed=20261024):
    y = np.asarray(labels)
    g = np.asarray(groups)
    _training_arrays(np.zeros((len(y), 1)), y)
    if g.ndim != 1 or len(g) != len(y) or any(not isinstance(v, str) or not v for v in groups):
        raise ValueError('one nonempty lineage string per observation required')
    if len(set(groups)) < n_splits:
        raise ValueError('insufficient independent groups for fixed folds')
    folds = np.full(len(y), -1, dtype=int)
    for index, (train, validation) in enumerate(StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y, g)):
        if set(g[train]) & set(g[validation]):
            raise ValueError('lineage leakage')
        if set(y[train]) != {0, 1} or set(y[validation]) != {0, 1}:
            raise ValueError('one-class fold; do not silently skip')
        if (folds[validation] >= 0).any():
            raise ValueError('duplicate validation membership')
        folds[validation] = index
    if (folds < 0).any():
        raise ValueError('missing out-of-fold membership')
    return folds


def fit_candidate(x, labels, kind):
    x, y = _training_arrays(x, labels)
    if kind not in KINDS or x.shape[1] != (1 if kind == 'R1' else 9):
        raise ValueError('candidate feature width mismatch')
    if kind == 'R2':
        scaler = StandardScaler().fit(x)
        mean, scale = scaler.mean_, scaler.scale_
    else:
        mean, scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    z = (x-mean)/scale
    model = LogisticRegression(C=1, class_weight='balanced' if kind == 'R2' else None,
                               fit_intercept=True, solver='lbfgs', max_iter=2000, tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(z, y)
    if kind == 'R1' and model.coef_[0, 0] <= 0:
        raise ValueError('R1 nonpositive slope fails monotone-calibration hypothesis')
    return {'schema_version': 1, 'kind': kind, 'research_only': True,
            'train_n': len(y), 'train_ai_n': int(y.sum()),
            'mean': mean.tolist(), 'scale': scale.tolist(),
            'coefficient': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'iterations': int(model.n_iter_[0]), 'C': 1,
            'class_weight': 'balanced' if kind == 'R2' else None}


def predict_candidate(model, x):
    if model.get('schema_version') != 1 or model.get('kind') not in KINDS or model.get('research_only') is not True:
        raise ValueError('invalid research model schema')
    width = 1 if model['kind'] == 'R1' else 9
    arrays = [np.asarray(model[k], dtype=float) for k in ('mean', 'scale', 'coefficient')]
    if any(a.shape != (width,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('invalid model feature parameters')
    mean, scale, coef = arrays
    if (scale <= 0).any() or not np.isfinite(model['intercept']):
        raise ValueError('invalid scale/intercept')
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != width or not np.isfinite(x).all():
        raise ValueError('invalid prediction features')
    return expit(((x-mean)/scale) @ coef + model['intercept'])


def binary_metrics(labels, verdicts):
    """Unknown/error labels stay unsuccessful and uncovered, never removed."""
    y = np.asarray(labels)
    v = np.asarray(verdicts)
    if y.ndim != 1 or len(y) == 0 or y.dtype.kind not in 'iu' or not set(y.tolist()) <= {0, 1}:
        raise ValueError('binary integer labels required')
    if v.ndim != 1 or len(v) != len(y) or not set(v.tolist()) <= {'Human', 'AI', 'Mixed', 'Uncertain', 'Error'}:
        raise ValueError('invalid verdicts or lengths')
    h, a = y == 0, y == 1
    flag = np.isin(v, ['AI', 'Mixed'])
    accepted = np.isin(v, ['Human', 'AI', 'Mixed'])
    correct = (h & (v == 'Human')) | (a & flag)
    def ratio(num, den):
        return float(num/den) if den else None
    return {'n': len(y), 'human_n': int(h.sum()), 'ai_n': int(a.sum()),
            'human_false_flags': int((h & flag).sum()), 'human_correct': int((h & (v == 'Human')).sum()),
            'ai_correct': int((a & flag).sum()), 'correct': int(correct.sum()),
            'accepted': int(accepted.sum()), 'uncertain': int((v == 'Uncertain').sum()),
            'errors': int((v == 'Error').sum()), 'human_fpr': ratio((h & flag).sum(), h.sum()),
            'ai_recall': ratio((a & flag).sum(), a.sum()),
            'failure_inclusive_binary_correctness': float(correct.mean()), 'coverage': float(accepted.mean()),
            'balanced_coverage': (float(accepted[h].mean())+float(accepted[a].mean()))/2 if h.any() and a.any() else None,
            'selective_correctness': ratio(correct.sum(), accepted.sum()),
            'accepted_human_fpr': ratio((h & flag).sum(), (h & accepted).sum()),
            'accepted_ai_recall': ratio((a & flag).sum(), (a & accepted).sum())}
