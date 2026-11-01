"""Train an experimental EEG classifier from a public dataset (stress/mental load or depression).

Examples:
  python train_public_eeg.py --dataset eegmat --root data/eegmat --output-dir eeg_model_stress
  python train_public_eeg.py --dataset mumtaz --root data/mumtaz --condition EC --output-dir eeg_model_depression_offline

The output directory must not exist. Stress artifacts can be loaded by the live dashboard with
BHM_EEG_MODEL_DIR; depression artifacts are offline research baselines and are never loaded live.
See docs/EEG_PUBLIC_DATA.md.
"""
import argparse
import json
from pathlib import Path
from app.eeg.public_data import load_eegmat, load_mumtaz, load_recordings_manifest
from app.eeg.public_training import train_public

DATASETS = {
    'eegmat': dict(task='Rest versus mental arithmetic (cognitive load: stress and focused effort combined)',
                   module_id='stress_state_index', live_compatible=True,
                   provenance='PhysioNet "EEG During Mental Arithmetic Tasks" (Zyma et al. 2019)',
                   license_text='Open Data Commons Attribution v1.0 (verify on the PhysioNet page)'),
    'mumtaz': dict(task='Major depressive disorder versus healthy controls (offline research baseline)',
                   module_id='depression_screening_module', live_compatible=False,
                   provenance='Mumtaz et al. MDD/healthy-control resting EEG (figshare)',
                   license_text='Check the dataset page; cite the original authors'),
    'manifest': dict(task=None, module_id='custom_module', live_compatible=False,
                     provenance='User-converted public recordings', license_text='User must verify dataset license'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    p.add_argument('--root', type=Path, help='Folder containing the downloaded EDF files (eegmat, mumtaz)')
    p.add_argument('--manifest', type=Path, help='JSON manifest of .npy recordings (dataset=manifest)')
    p.add_argument('--task-name', help='Task description for dataset=manifest')
    p.add_argument('--channel', default='Fp1', help='Single channel to use, e.g. Fp1 (forehead, closest to the BioAmp placement)')
    p.add_argument('--condition', default='EC', help='mumtaz only: EC, EO or TASK')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-robust-z', type=float, default=12.0, help='Reject windows with outliers beyond this many robust sigmas')
    p.add_argument('--cv-folds', type=int, default=5)
    args = p.parse_args()

    spec = dict(DATASETS[args.dataset])
    if args.dataset == 'eegmat':
        recordings = load_eegmat(args.root, args.channel)
    elif args.dataset == 'mumtaz':
        recordings = load_mumtaz(args.root, args.channel, args.condition)
    else:
        if not args.manifest or not args.task_name:
            p.error('--manifest and --task-name are required for dataset=manifest')
        recordings = load_recordings_manifest(args.manifest)
        spec['task'] = args.task_name
    if args.dataset != 'manifest' and not args.root:
        p.error('--root is required for this dataset')

    report = train_public(recordings, args.output_dir, dataset=args.dataset, channel=args.channel,
                          seed=args.seed, max_robust_z=args.max_robust_z, cv_folds=args.cv_folds, **spec)
    summary = {'selected': report['selected'], 'splits': report['splits'],
               'validation_balanced_accuracy': {k: v['balanced_accuracy'] for k, v in report['validation'].items()},
               'test': {k: report['test'][k] for k in ('accuracy', 'balanced_accuracy', 'macro_f1', 'roc_auc',
                                                       'balanced_accuracy_ci95_subject_bootstrap')},
               'test_dummy_baseline_balanced_accuracy': report['baseline_test']['balanced_accuracy'],
               'subject_cv_selected_model': report['subject_cv_selected_model'],
               'subject_cv_dummy': report['subject_cv_dummy'],
               'rejected_windows': len(report['rejected_windows'])}
    print(json.dumps(summary, indent=2))
    print(f'\nArtifact written to {args.output_dir}')


if __name__ == '__main__':
    main()
