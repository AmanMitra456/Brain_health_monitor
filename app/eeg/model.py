"""Load trusted local sklearn artifacts only; never accept uploads/pickle paths.

Artifact schemas:
  1: all FEATURE_NAMES, raw ADC counts recorded on this hardware (original contract).
  2: ordered subset of FEATURE_NAMES; units 'adc_counts' or 'scale_free'. Scale-free artifacts are
     trained on public data after resampling and use only amplitude-independent features.
     `live_compatible: false` marks offline research artifacts that must never run on live input.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import sklearn
from .signal import FEATURE_NAMES, SignalConfig

SCALE_FREE_LIMITATION = ('Experimental classifier trained on public multichannel data after single-channel resampling; '
                         'transfer to this Maker UNO/BioAmp hardware is unvalidated and probabilities are not calibrated')
DEFAULT_LIMITATION = 'Experimental task classifier; probabilities are not calibrated reliability or diagnosis'


class OfflineArtifact(ValueError):
    """Artifact is valid but explicitly marked as not usable for live inference."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _artifact_features(meta):
    """Validate the schema/units contract and return the ordered feature names the model expects."""
    version = meta['schema_version']
    if version == 1:
        if meta['feature_names'] != list(FEATURE_NAMES) or meta['units'] != 'adc_counts':
            raise ValueError('Incompatible feature schema or units')
        return list(FEATURE_NAMES)
    if version == 2:
        names = list(meta['feature_names'])
        if not names or names != [n for n in FEATURE_NAMES if n in names]:
            raise ValueError('Features must be a non-empty ordered subset of the live feature set')
        if meta['units'] not in ('adc_counts', 'scale_free'):
            raise ValueError('Unsupported units')
        if not meta.get('live_compatible', True):
            raise OfflineArtifact('Offline research artifact; not enabled for live inference')
        return names
    raise ValueError('Unsupported artifact schema')


class EEGModel:
    def __init__(self, directory):
        self.estimator = None
        self.feature_names = list(FEATURE_NAMES)
        self._indices = list(range(len(FEATURE_NAMES)))
        self.info = {'available': False, 'reason': 'No compatible EEG artifact installed; labeled subject-wise data required'}
        directory = Path(directory)
        try:
            meta = json.loads((directory/'metadata.json').read_text())
            names = _artifact_features(meta)
            SignalConfig(**meta['preprocessing'])
            if meta['sklearn_version'] != sklearn.__version__ or meta['python_version'] != platform.python_version():
                raise ValueError('Artifact dependency versions differ; retrain in this environment')
            if meta['model_sha256'] != digest(directory/'model.joblib'):
                raise ValueError('EEG artifact checksum mismatch')
            estimator = joblib.load(directory/'model.joblib')
            if list(estimator.classes_) != meta['labels'] or estimator.n_features_in_ != len(names):
                raise ValueError('EEG model labels or feature dimensions mismatch')
            self.estimator = estimator
            self.feature_names = names
            self._indices = [FEATURE_NAMES.index(n) for n in names]
            limitation = SCALE_FREE_LIMITATION if meta['units'] == 'scale_free' else DEFAULT_LIMITATION
            self.info = {**meta, 'available': True, 'limitation': meta.get('limitation', limitation)}
        except OfflineArtifact as exc:
            self.info['reason'] = str(exc)
        except Exception:
            # Error text deliberately omits local paths and pickle exception detail.
            self.info['reason'] = 'EEG model absent or incompatible; inspect artifact and environment'

    def predict(self, analysis, config):
        if self.estimator is None:
            return {'available': False, 'reason': self.info['reason']}
        if not analysis.get('valid') or asdict(config) != self.info['preprocessing'] or analysis.get('feature_names') != list(FEATURE_NAMES):
            return {'available': False, 'reason': 'Signal or preprocessing differs from training contract'}
        x = np.asarray(analysis['features'], dtype=float)
        if x.shape != (len(FEATURE_NAMES),) or not np.isfinite(x).all():
            return {'available': False, 'reason': 'Features unavailable or invalid'}
        x = x[self._indices]
        probabilities = self.estimator.predict_proba(x[None])[0]
        index = int(np.argmax(probabilities))
        return {'available': True, 'label': str(self.estimator.classes_[index]),
                'probabilities': {str(k): float(v) for k, v in zip(self.estimator.classes_, probabilities)},
                'score': float(probabilities[index]), 'model_version': self.info['version'],
                'task': self.info['task'], 'limitation': self.info['limitation']}
