import json
import pytest
from prepare_eeg_recordings import prepare


def export(path, source='serial', times=None):
    times = times or [0.0, 0.004, 0.008, 0.012]
    lines = ['source,sequence,device_seconds,adc_counts,received_utc']
    lines += [f'{source},{i},{t},{500 + i},2026-11-02T00:00:00Z' for i, t in enumerate(times)]
    path.write_text('\n'.join(lines) + '\n')
    return path


def test_converts_serial_export_and_leaves_split_unassigned(tmp_path):
    target = prepare(export(tmp_path / 'e.csv'), tmp_path / 'data', 's01', 'focus', tmp_path / 'data' / 'manifest.json')
    assert target.read_text().splitlines()[:2] == ['timestamp_s,value', '0.0,500.0']
    manifest = json.loads((tmp_path / 'data' / 'manifest.json').read_text())
    assert manifest['units'] == 'adc_counts' and manifest['recordings'][0]['split'] == 'unassigned'
    assert manifest['recordings'][0]['label'] == 'focus'


def test_refuses_simulated_nonmonotonic_and_overwrite(tmp_path):
    with pytest.raises(ValueError, match='real serial'):
        prepare(export(tmp_path / 'sim.csv', source='simulated'), tmp_path / 'd', 's1', 'rest', tmp_path / 'd' / 'm.json')
    with pytest.raises(ValueError, match='increasing'):
        prepare(export(tmp_path / 'bad.csv', times=[0.0, 0.004, 0.004, 0.012]), tmp_path / 'd', 's1', 'rest', tmp_path / 'd' / 'm.json')
    good = export(tmp_path / 'ok.csv')
    prepare(good, tmp_path / 'd', 's1', 'rest', tmp_path / 'd' / 'm.json')
    with pytest.raises(ValueError, match='overwrite'):
        prepare(good, tmp_path / 'd', 's1', 'rest', tmp_path / 'd' / 'm.json')


def test_unassigned_split_is_rejected_by_trainer(tmp_path):
    from app.eeg.training import load_manifest
    prepare(export(tmp_path / 'e.csv'), tmp_path / 'data', 's01', 'focus', tmp_path / 'data' / 'manifest.json')
    manifest = json.loads((tmp_path / 'data' / 'manifest.json').read_text())
    manifest.update(task='t', provenance='p', license='l')
    (tmp_path / 'data' / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='explicit split'):
        load_manifest(tmp_path / 'data' / 'manifest.json')
