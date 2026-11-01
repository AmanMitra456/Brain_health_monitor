"""Turn a CSV exported from the /eeg dashboard into a labeled training recording.

The dashboard export has columns `source,sequence,device_seconds,adc_counts,received_utc`. The trainer
(`train_eeg.py`) needs `timestamp_s,value`, only real serial data, and an explicit manifest entry.
This script does that conversion, refuses simulated data and non-increasing device clocks, and adds
the recording to a manifest with split "unassigned": you must then set train/validation/test per
SUBJECT yourself (a subject may only appear in one split) before training.

  python prepare_eeg_recordings.py --export eeg-123.csv --subject s01 --label focus \
      --out-dir my_eeg_data --manifest my_eeg_data/manifest.json
"""
import argparse
import csv
import json
from pathlib import Path


def prepare(export_csv, out_dir, subject, label, manifest_path):
    export_csv, out_dir, manifest_path = Path(export_csv), Path(out_dir), Path(manifest_path)
    with export_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        needed = {'source', 'device_seconds', 'adc_counts'}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError(f'Export must contain columns {sorted(needed)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Export contains no samples')
    sources = {r['source'] for r in rows}
    if sources != {'serial'}:
        raise ValueError(f'Only real serial recordings can be used for training; found sources {sorted(sources)}')
    times = [float(r['device_seconds']) for r in rows]
    values = [float(r['adc_counts']) for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Device timestamps are not strictly increasing; re-record instead of repairing')
    if not subject.strip() or not label.strip():
        raise ValueError('Subject and label are required')

    out_dir.mkdir(parents=True, exist_ok=True)
    name = f'{subject}_{label}_{export_csv.stem}.csv'
    target = out_dir / name
    if target.exists():
        raise ValueError(f'{target} already exists; refusing to overwrite')
    with target.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_s', 'value'])
        writer.writerows(zip(times, values))

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {'task': 'FILL IN: the labeled protocol, e.g. relax vs focus vs stress', 'provenance': 'FILL IN: investigator, protocol version, date',
                    'license': 'FILL IN: consent and permitted use', 'units': 'adc_counts', 'sample_rate': 250, 'recordings': []}
    try:
        relative = str(target.relative_to(manifest_path.parent))
    except ValueError:
        relative = str(target.resolve())
    manifest['recordings'].append({'path': relative, 'subject': subject, 'label': label, 'split': 'unassigned'})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--subject', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    target = prepare(args.export, args.out_dir, args.subject, args.label, args.manifest)
    print(f'Wrote {target} and updated {args.manifest}. Set each recording split before running train_eeg.py.')


if __name__ == '__main__':
    main()
