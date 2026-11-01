"""Subject-disjoint training of scale-free single-channel EEG classifiers from public recordings.

Same windows, filters and feature code as live inference (`signal._spectral_core`), restricted to
amplitude-independent features. Splits are by SUBJECT before windowing, windows never overlap, the
model is selected on validation balanced accuracy only and the held-out test set is scored once.
Because public datasets have few subjects, subject-wise cross-validation and a subject-level
bootstrap interval are reported next to the single test split; treat all numbers as development
estimates, not evidence of performance on the Maker UNO/BioAmp hardware.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .model import digest
from .public_data import recording_windows
from .signal import SCALE_FREE_FEATURES, SignalConfig
from .storage import utc_now
from .training import metrics, write_confusion_svg

OFFLINE_LIMITATION = ('Offline research baseline on public data: single-channel features cannot replace '
                      'bilateral frontal asymmetry, so this must not be used as a live or diagnostic tool')


def subject_split(subject_labels, seed=42, fractions=(.6, .2, .2)):
    """Assign whole subjects to train/validation/test, stratified by the set of labels they carry.

    `subject_labels` maps subject -> iterable of labels. Subjects that carry the same label set
    (for example every healthy control) are shuffled and divided separately so each split keeps
    every class. Needs at least three subjects per group.
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for subject, labels in sorted(subject_labels.items()):
        groups.setdefault(tuple(sorted(labels)), []).append(subject)
    assignment = {}
    for key, subjects in groups.items():
        if len(subjects) < 3:
            raise ValueError(f'Need at least 3 subjects with labels {key} to hold out validation and test; found {len(subjects)}')
        order = list(subjects)
        rng.shuffle(order)
        n = len(order)
        n_val, n_test = max(1, round(n * fractions[1])), max(1, round(n * fractions[2]))
        n_train = n - n_val - n_test
        if n_train < 1:
            raise ValueError('Too few subjects for the requested split fractions')
        for i, subject in enumerate(order):
            assignment[subject] = 'train' if i < n_train else 'validation' if i < n_train + n_val else 'test'
    return assignment


def subject_bootstrap_ci(predicted, y, groups, resamples=1000, seed=0):
    """95% interval of balanced accuracy when whole subjects are resampled (windows are correlated within subject)."""
    rng = np.random.default_rng(seed)
    subjects = sorted(set(groups))
    stats = []
    for _ in range(resamples):
        picked = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == s) for s in picked])
        if len(set(y[idx])) < 2:
            continue
        stats.append(balanced_accuracy_score(y[idx], predicted[idx]))
    if len(stats) < 50:
        return None
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def grouped_cv(model, x, y, groups, folds):
    """Subject-wise cross-validation: every fold tests on subjects never seen in that fold's training."""
    n_splits = min(folds, len(set(groups)))
    if n_splits < 2:
        return None
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
            continue  # a fold without both classes cannot be scored with balanced accuracy
        fitted = clone(model).fit(x[train_idx], y[train_idx])
        scores.append(balanced_accuracy_score(y[test_idx], fitted.predict(x[test_idx])))
    if not scores:
        return None
    return {'folds_scored': len(scores), 'mean_balanced_accuracy': float(np.mean(scores)),
            'std_balanced_accuracy': float(np.std(scores)), 'fold_scores': [float(s) for s in scores]}


def train_public(recordings, output_dir, *, task, module_id, dataset, channel, provenance, license_text,
                 live_compatible=True, config=None, seed=42, max_robust_z=12.0, cv_folds=5):
    out = Path(output_dir)
    if out.exists():
        raise ValueError('Use a new output directory to preserve prior model/evaluation')
    if not recordings:
        raise ValueError('No recordings supplied')
    config = config or SignalConfig()

    rows, labels, subjects, rejected, audit = [], [], [], [], []
    for rec in recordings:
        accepted, bad = recording_windows(rec, config, max_robust_z)
        rows += accepted
        labels += [rec.label] * len(accepted)
        subjects += [rec.subject] * len(accepted)
        rejected += bad
        audit.append({'recording': rec.source, 'subject': rec.subject, 'label': rec.label, 'condition': rec.condition,
                      'original_rate_hz': rec.fs, 'duration_seconds': float(len(rec.values) / rec.fs),
                      'accepted_windows': len(accepted), 'rejected_windows': len(bad),
                      'sha256': hashlib.sha256(np.asarray(rec.values, '<f8').tobytes()).hexdigest()})
    if not rows:
        raise ValueError('No usable windows after quality screening')
    x, y, g = np.asarray(rows, dtype=float), np.asarray(labels), np.asarray(subjects)
    label_list = sorted(set(labels))
    if len(label_list) < 2:
        raise ValueError('Need at least two classes')
    if len({r['sha256'] for r in audit}) != len(audit):
        raise ValueError('Duplicate recording content; remove repeated recordings')

    assignment = subject_split({s: set(y[g == s]) for s in sorted(set(g))}, seed)
    split_of = np.array([assignment[s] for s in g])
    data = {s: (x[split_of == s], y[split_of == s], g[split_of == s]) for s in ('train', 'validation', 'test')}
    for split, (_, ys, _) in data.items():
        if set(ys) != set(label_list):
            raise ValueError(f'{split} split lacks windows from every class; add subjects or relax quality rejection')

    candidates = {'dummy': DummyClassifier(strategy='prior'),
                  'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
                  'random_forest': RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=2, class_weight='balanced', random_state=seed, n_jobs=1)}
    validation = {}
    for name, model in candidates.items():
        model.fit(*data['train'][:2])
        validation[name] = metrics(model, data['validation'][0], data['validation'][1], label_list)
    selected = max(candidates, key=lambda n: validation[n]['balanced_accuracy'])
    model = candidates[selected]

    x_test, y_test, g_test = data['test']
    test = metrics(model, x_test, y_test, label_list)
    test['balanced_accuracy_ci95_subject_bootstrap'] = subject_bootstrap_ci(model.predict(x_test), y_test, g_test, seed=seed)
    report = {
        'selected': selected, 'selection_metric': 'validation balanced_accuracy',
        'validation': validation, 'test': test,
        'baseline_test': metrics(candidates['dummy'], x_test, y_test, label_list),
        'subject_cv_selected_model': grouped_cv(model, x, y, g, cv_folds),
        'subject_cv_dummy': grouped_cv(candidates['dummy'], x, y, g, cv_folds),
        'splits': {s: {'windows': len(d[1]), 'subjects': len(set(d[2])),
                       'class_counts': {l: int(np.sum(d[1] == l)) for l in label_list}} for s, d in data.items()},
        'recording_audit': audit, 'rejected_windows': rejected,
        'limitation': ('Subject-disjoint, non-overlapping windows from a public dataset. Window scores are correlated within '
                       'subject, the interval is wide with few subjects, and transfer to Maker UNO/BioAmp recordings is unvalidated.')}

    out.mkdir(parents=True)
    joblib.dump(model, out / 'model.joblib')
    meta = {'schema_version': 2, 'version': utc_now(), 'name': selected, 'task': task, 'module_id': module_id,
            'units': 'scale_free', 'feature_names': list(SCALE_FREE_FEATURES), 'labels': label_list,
            'preprocessing': asdict(config), 'live_compatible': bool(live_compatible),
            'training_domain': {'dataset': dataset, 'channel': channel, 'resampled_to_hz': config.sample_rate,
                                'amplitude': 'public dataset native unit; only scale-free features used'},
            'model_sha256': digest(out / 'model.joblib'), 'sklearn_version': sklearn.__version__,
            'python_version': platform.python_version(), 'provenance': provenance, 'license': license_text,
            'dependencies': {'numpy': np.__version__, 'scipy': scipy.__version__, 'joblib': joblib.__version__},
            'training': {'seed': seed, 'selection': 'validation balanced_accuracy', 'max_robust_z': max_robust_z,
                         'window_hop_samples': config.input_samples}}
    if not live_compatible:
        meta['limitation'] = OFFLINE_LIMITATION
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
    (out / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / 'manifest.json').write_text(json.dumps({'dataset': dataset, 'channel': channel, 'provenance': provenance,
                                                   'license': license_text, 'subject_splits': assignment,
                                                   'recordings': audit}, indent=2))
    write_confusion_svg(out / 'confusion_matrix.svg', report['test'])
    return report
