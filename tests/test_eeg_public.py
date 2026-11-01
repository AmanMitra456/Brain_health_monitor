"""Tests for the public-dataset EEG path.

All signals here are SYNTHETIC and exist only to exercise software behaviour (splits, leakage guards,
artifact contract, scale-free equivalence with live analysis). They say nothing about real EEG accuracy.
"""
import numpy as np
import pytest
from app.eeg.model import EEGModel
from app.eeg.public_data import (Recording, load_eegmat, load_mumtaz, load_recordings_manifest,
                                 recording_windows, resample_to)
from app.eeg.public_training import subject_split, train_public
from app.eeg.signal import FEATURE_NAMES, SCALE_FREE_FEATURES, SignalConfig, analyze, window_features

CONFIG = SignalConfig()


def synth(rng, seconds, fs, alpha, beta, gain=1.0):
    """Noise plus alpha (10 Hz) and beta (20 Hz) rhythms; synthetic only."""
    t = np.arange(int(seconds * fs)) / fs
    x = rng.normal(0, 1.0, len(t))
    x += alpha * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6.28))
    x += beta * np.sin(2 * np.pi * 20 * t + rng.uniform(0, 6.28))
    return gain * x


def synthetic_recordings(n_subjects=16, fs=500, seconds=90, seed=1):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_subjects):
        gain = rng.uniform(0.5, 40)  # per-subject amplitude scale must not matter
        jitter = rng.uniform(0.8, 1.2)
        recs.append(Recording(f'S{i:02d}', 'rest', synth(rng, seconds, fs, 3.0 * jitter, 0.3, gain), fs, f's{i}_1.edf'))
        recs.append(Recording(f'S{i:02d}', 'mental_task', synth(rng, seconds, fs, 0.8 * jitter, 1.6, gain), fs, f's{i}_2.edf'))
    return recs


def train(tmp_path, name='art', recs=None, **kw):
    args = dict(task='synthetic rest vs task', module_id='stress_state_index', dataset='synthetic', channel='Fp1',
                provenance='synthetic test data', license_text='n/a', cv_folds=4)
    args.update(kw)
    return train_public(recs or synthetic_recordings(), tmp_path / name, **args)


def test_scale_free_features_ignore_amplitude_and_match_live_analysis():
    rng = np.random.default_rng(0)
    x = synth(rng, 6, 250, 3.0, 1.0)
    base, reason = window_features(x, CONFIG)
    scaled, _ = window_features(x * 37.5 + 1000, CONFIG)
    assert reason is None
    for name in SCALE_FREE_FEATURES:
        assert scaled[name] == pytest.approx(base[name], rel=1e-6)
    # Same window expressed as ADC counts through the live analyze(): identical shared feature math.
    adc = 512 + 12 * x
    t = np.arange(len(adc)) / 250
    live = analyze(t, adc, CONFIG)
    assert live['valid'], live['reasons']
    for name in SCALE_FREE_FEATURES:
        assert live['features'][FEATURE_NAMES.index(name)] == pytest.approx(base[name], rel=1e-6)
    assert 'variance' not in SCALE_FREE_FEATURES


@pytest.mark.parametrize('kind,reason', [('flat', 'flatline'), ('spike', 'amplitude_outlier'), ('nan', 'non_finite_samples')])
def test_window_quality_rejections(kind, reason):
    x = synth(np.random.default_rng(2), 6, 250, 3.0, 1.0)
    if kind == 'flat':
        x[:] = 5.0
    elif kind == 'spike':
        x[700] += 500
    else:
        x[10] = np.nan
    features, got = window_features(x, CONFIG)
    assert features is None and got == reason


def test_resampling_to_hardware_rate_and_window_cutting():
    assert len(resample_to(np.zeros(5000), 500, 250)) == 2500
    assert len(resample_to(np.zeros(1280), 128, 250)) == 2500
    rec = Recording('S01', 'rest', synth(np.random.default_rng(3), 20, 500, 3, 1), 500, 'x.edf')
    rows, rejected = recording_windows(rec, CONFIG)
    assert len(rows) == 3 and not rejected  # 20 s -> three 6 s windows, 2 s tail discarded
    assert len(rows[0]) == len(SCALE_FREE_FEATURES)


def test_subject_split_is_disjoint_stratified_and_deterministic():
    subjects = {f'H{i}': {'healthy'} for i in range(10)} | {f'M{i}': {'depressed'} for i in range(10)}
    a = subject_split(subjects, seed=7)
    assert a == subject_split(subjects, seed=7)
    for split in ('train', 'validation', 'test'):
        members = [s for s, v in a.items() if v == split]
        assert any(s.startswith('H') for s in members) and any(s.startswith('M') for s in members)
    assert len(a) == 20
    with pytest.raises(ValueError):
        subject_split({'a': {'x'}, 'b': {'x'}}, seed=1)


def test_training_roundtrip_live_prediction_and_contract(tmp_path):
    report = train(tmp_path)
    # Strong synthetic effect: only checks that the pipeline can learn and beats the prior baseline.
    assert report['test']['balanced_accuracy'] > report['baseline_test']['balanced_accuracy'] + 0.2
    assert report['subject_cv_selected_model']['folds_scored'] >= 2
    splits = report['splits']
    assert sum(s['subjects'] for s in splits.values()) == 16
    out = tmp_path / 'art'
    model = EEGModel(out)
    assert model.info['available'] and model.info['units'] == 'scale_free'
    assert model.feature_names == list(SCALE_FREE_FEATURES)
    # Live window in ADC counts, scored through the normal analyze() -> predict() route.
    rng = np.random.default_rng(9)
    adc = 512 + 10 * synth(rng, 6, 250, 0.8, 1.6)
    analysis = analyze(np.arange(len(adc)) / 250, adc, CONFIG)
    prediction = model.predict(analysis, CONFIG)
    assert prediction['available'] and prediction['label'] in ('rest', 'mental_task')
    assert 'unvalidated' in prediction['limitation']
    # Changing the preprocessing contract disables prediction instead of guessing.
    other = SignalConfig(notch_hz=60.)
    assert not model.predict(analysis, other)['available']


def test_offline_artifact_is_never_loaded_for_live_use(tmp_path):
    train(tmp_path, live_compatible=False, module_id='depression_screening_module')
    model = EEGModel(tmp_path / 'art')
    assert not model.info['available'] and 'Offline' in model.info['reason']


def test_leakage_and_overwrite_guards(tmp_path):
    recs = synthetic_recordings(n_subjects=6)
    train(tmp_path, recs=recs)
    with pytest.raises(ValueError, match='new output directory'):
        train(tmp_path, recs=recs)
    dup = synthetic_recordings(n_subjects=6)
    dup.append(Recording('S99', 'rest', dup[0].values.copy(), dup[0].fs, 'copy.edf'))
    with pytest.raises(ValueError, match='Duplicate'):
        train(tmp_path, name='dup', recs=dup)
    with pytest.raises(ValueError):  # too few subjects to hold out validation and test
        train(tmp_path, name='few', recs=synthetic_recordings(n_subjects=2))


def fake_reader_factory(fs=500):
    rng = np.random.default_rng(4)

    def reader(path, channel):
        reader.calls.append((path.name, channel))
        return synth(rng, 12, fs, 2.0, 0.5), float(fs)
    reader.calls = []
    return reader


def test_eegmat_loader_labels_and_subjects(tmp_path):
    for n in (0, 1):
        for part in (1, 2):
            (tmp_path / f'Subject{n:02d}_{part}.edf').write_bytes(b'')
    (tmp_path / 'subject-info.csv').write_text('x')
    reader = fake_reader_factory()
    recs = load_eegmat(tmp_path, 'Fp1', reader=reader)
    assert sorted((r.subject, r.label) for r in recs) == [('S00', 'mental_task'), ('S00', 'rest'), ('S01', 'mental_task'), ('S01', 'rest')]
    assert all(c[1] == 'Fp1' for c in reader.calls)
    with pytest.raises(FileNotFoundError):
        load_eegmat(tmp_path / 'missing', reader=reader)


def test_mumtaz_loader_filters_condition_and_maps_groups(tmp_path):
    for name in ('MDD S1 EC.edf', 'H S10 EC.edf', 'H S1 EO.edf', 'MDD S2 TASK.edf', 'notes.txt'):
        (tmp_path / name).write_bytes(b'')
    recs = load_mumtaz(tmp_path, 'Fp1', 'EC', reader=fake_reader_factory())
    assert sorted((r.subject, r.label) for r in recs) == [('H10', 'healthy'), ('MDD01', 'depressed')]
    with pytest.raises(FileNotFoundError, match='saw'):
        load_mumtaz(tmp_path, 'Fp1', 'XX', reader=fake_reader_factory())


def test_manifest_loader_roundtrip_and_validation(tmp_path):
    np.save(tmp_path / 'a.npy', synth(np.random.default_rng(5), 10, 128, 2, 1))
    (tmp_path / 'm.json').write_text('{"recordings":[{"path":"a.npy","subject":"s1","label":"rest","fs":128}]}')
    recs = load_recordings_manifest(tmp_path / 'm.json')
    assert recs[0].fs == 128 and recs[0].label == 'rest'
    (tmp_path / 'bad.json').write_text('{"recordings":[{"path":"a.npy","subject":"s1"}]}')
    with pytest.raises(ValueError):
        load_recordings_manifest(tmp_path / 'bad.json')


def registry_for(tmp_path, artifact):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.eeg.api import router
    from app.eeg.service import EEGService
    from app.eeg.storage import RecordingStore
    app = FastAPI()
    app.include_router(router)
    app.state.eeg = EEGService(RecordingStore(tmp_path / 'eeg.db'), SignalConfig(), EEGModel(artifact))
    with TestClient(app) as client:
        yield client
    app.state.eeg.disconnect()


def test_registry_enables_only_the_declared_module(tmp_path):
    train(tmp_path)
    for client in registry_for(tmp_path, tmp_path / 'art'):
        registry = client.get('/api/eeg/models/registry').json()
        available = {m['module_id'] for m in registry['modules'] if m['availability']}
        assert available == {'stress_state_index'}
        stress = next(m for m in registry['modules'] if m['module_id'] == 'stress_state_index')
        assert stress['version'] and 'unvalidated' in stress['limitations']
        info = client.get('/api/eeg/model/info').json()
        assert info['module_id'] == 'stress_state_index' and info['units'] == 'scale_free'


def test_registry_ignores_offline_depression_artifact(tmp_path):
    train(tmp_path, live_compatible=False, module_id='depression_screening_module')
    for client in registry_for(tmp_path, tmp_path / 'art'):
        registry = client.get('/api/eeg/models/registry').json()
        assert not any(m['availability'] for m in registry['modules'])
        assert registry['active_model'] is None


def test_real_edf_reader_matches_channel_names(tmp_path):
    """Exercises MNE's EDF parsing with a synthetic file; skipped when optional dependencies are absent."""
    mne = pytest.importorskip('mne')
    pytest.importorskip('edfio')
    from app.eeg.public_data import read_edf_channel
    rng = np.random.default_rng(0)
    names = ['EEG Fp1', 'EEG Fz', 'ECG ECG']
    data = np.vstack([synth(rng, 10, 500, 3.0, 1.0) * 1e-6 for _ in names])
    info = mne.create_info(names, 500., ch_types=['eeg', 'eeg', 'ecg'])
    path = tmp_path / 'Subject00_1.edf'
    mne.export.export_raw(str(path), mne.io.RawArray(data, info, verbose='ERROR'), fmt='edf', overwrite=True, verbose='ERROR')
    values, fs = read_edf_channel(path, 'fp1')
    assert fs == 500.0 and len(values) == 5000 and 1.0 < values.std() < 5.0
    with pytest.raises(ValueError, match='available'):
        read_edf_channel(path, 'O9')
