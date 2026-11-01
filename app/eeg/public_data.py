"""Adapters that turn public EEG recordings into windows this project's pipeline can use.

Public datasets are multichannel, in microvolts and recorded at various rates. The live device is one
channel of uncalibrated 10-bit ADC counts at 250 Hz. These adapters therefore pick ONE channel,
resample to the configured rate and compute only the scale-free subset of the shared features
(see `signal.SCALE_FREE_FEATURES`). Nothing here converts microvolts into "ADC counts".

EDF parsing is delegated to MNE-Python (optional dependency, imported lazily).
"""
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import numpy as np
from scipy.signal import resample_poly
from .signal import SCALE_FREE_FEATURES, segment, window_features


@dataclass(frozen=True)
class Recording:
    subject: str
    label: str
    values: np.ndarray  # one channel, original amplitude unit
    fs: float           # original sampling rate in Hz
    source: str         # file the data came from (for audit/provenance)
    condition: str = ''


def _norm_channel(name):
    """'EEG Fp1', 'EEG Fp1-LE' and 'fp1' all normalise to 'FP1'."""
    n = name.strip()
    if n.upper().startswith('EEG'):
        n = n[3:].strip()
    return re.split(r'[-_ ]', n)[0].upper()


def read_edf_channel(path, channel):
    """Return (samples, sampling_rate) for one named channel of an EDF file, using MNE-Python."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError('Reading EDF files needs MNE-Python: pip install -r requirements-eeg-data.txt') from exc
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose='ERROR')
    wanted = _norm_channel(channel)
    matches = [n for n in raw.ch_names if _norm_channel(n) == wanted]
    if len(matches) != 1:
        raise ValueError(f'Channel {channel!r} matched {len(matches)} channels in {Path(path).name}; available: {raw.ch_names}')
    data = raw.get_data(picks=matches)[0] * 1e6  # volts -> microvolts (scale does not affect scale-free features)
    return data, float(raw.info['sfreq'])


def resample_to(values, fs_in, fs_out):
    """Polyphase resampling with built-in anti-alias filtering."""
    values = np.asarray(values, dtype=float)
    if abs(fs_in - fs_out) < 1e-9:
        return values
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(values, ratio.numerator, ratio.denominator)


def recording_windows(recording, config, max_robust_z=12.0):
    """Cut one recording into non-overlapping live-sized windows.

    Returns (rows, rejected): rows are dicts of scale-free features, rejected is a list of
    {'source', 'start', 'reason'}. Incomplete tail samples are discarded, never padded.
    """
    x = resample_to(recording.values, recording.fs, config.sample_rate)
    rows, rejected = [], []
    for start, end in segment(len(x), config.input_samples, config.input_samples):
        features, reason = window_features(x[start:end], config, max_robust_z=max_robust_z)
        if features is None:
            rejected.append({'source': recording.source, 'start': int(start), 'reason': reason})
        else:
            rows.append([features[n] for n in SCALE_FREE_FEATURES])
    return rows, rejected


# --- dataset-specific loaders -------------------------------------------------------------------

EEGMAT_FILE = re.compile(r'^Subject(\d+)_([12])\.edf$', re.IGNORECASE)
EEGMAT_LABELS = {'1': 'rest', '2': 'mental_task'}


def load_eegmat(root, channel='Fp1', reader=read_edf_channel):
    """PhysioNet 'EEG During Mental Arithmetic Tasks'.

    SubjectNN_1.edf is background EEG before the task (label 'rest'); SubjectNN_2.edf is EEG during
    mental arithmetic (label 'mental_task'). 'mental_task' is cognitive load: it mixes stress and
    focused effort and cannot separate them.
    """
    root = Path(root)
    files = sorted(p for p in root.rglob('*.edf') if EEGMAT_FILE.match(p.name))
    if not files:
        raise FileNotFoundError(f'No SubjectNN_1/2.edf files under {root}; download the dataset first (docs/EEG_PUBLIC_DATA.md)')
    recordings = []
    for path in files:
        number, part = EEGMAT_FILE.match(path.name).groups()
        values, fs = reader(path, channel)
        recordings.append(Recording(subject=f'S{int(number):02d}', label=EEGMAT_LABELS[part], values=values,
                                    fs=fs, source=path.name, condition='arithmetic' if part == '2' else 'baseline'))
    return recordings


MUMTAZ_FILE = re.compile(r'^(H|MDD)\s*S?(\d+)\s*(EC|EO|TASK)\.edf$', re.IGNORECASE)
MUMTAZ_LABELS = {'H': 'healthy', 'MDD': 'depressed'}


def load_mumtaz(root, channel='Fp1', condition='EC', reader=read_edf_channel):
    """Mumtaz et al. MDD vs healthy-control EEG (files such as 'MDD S1 EC.edf', 'H S3 EO.edf').

    Labels are the dataset's clinical group labels. One subject has exactly one label.
    """
    root = Path(root)
    condition = condition.upper()
    candidates = sorted(p for p in root.rglob('*.edf'))
    files = [p for p in candidates if MUMTAZ_FILE.match(p.name) and MUMTAZ_FILE.match(p.name).group(3).upper() == condition]
    if not files:
        names = [p.name for p in candidates[:5]]
        raise FileNotFoundError(f'No {condition} recordings named like "MDD S1 {condition}.edf" under {root}; saw {names}')
    recordings = []
    for path in files:
        group, number, _ = MUMTAZ_FILE.match(path.name).groups()
        group = group.upper()
        values, fs = reader(path, channel)
        recordings.append(Recording(subject=f'{group}{int(number):02d}', label=MUMTAZ_LABELS[group], values=values,
                                    fs=fs, source=path.name, condition=condition))
    return recordings


def load_recordings_manifest(path):
    """Generic adapter for any dataset converted to .npy (1-D samples) files.

    JSON: {"recordings": [{"path": "s01_rest.npy", "subject": "s01", "label": "rest", "fs": 128}, ...]}
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    recordings = []
    for item in manifest.get('recordings', []):
        for field in ('path', 'subject', 'label', 'fs'):
            if field not in item:
                raise ValueError(f'Every recording needs {field}')
        source = (path.parent / item['path']).resolve()
        values = np.load(source)
        if values.ndim != 1:
            raise ValueError(f'{item["path"]} must be a one-dimensional single-channel array')
        recordings.append(Recording(subject=str(item['subject']), label=str(item['label']), values=values.astype(float),
                                    fs=float(item['fs']), source=item['path'], condition=str(item.get('condition', ''))))
    if not recordings:
        raise ValueError('Manifest lists no recordings')
    return recordings
