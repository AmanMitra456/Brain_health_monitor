# Training EEG models from public datasets

Use this path while no labeled recordings from the Maker UNO/BioAmp kit exist. It trains on public data,
adapted to this project's single-channel, 250 Hz pipeline. For recordings from your own kit see
[EEG_PROTOCOL.md](EEG_PROTOCOL.md) and [EEG_ML.md](EEG_ML.md).

## What this can and cannot claim

- **Stress / focus.** The public task used here is *rest versus mental arithmetic*. That measures cognitive
  load, a mix of stress and focused effort. It **cannot tell stress from focus**. Separating them needs
  recordings with both conditions and self-report, which is what the protocol document defines.
- **Depression.** One model for all three targets is not justified: stress/focus are short-lived states
  labeled by task, depression is a trait labeled by clinical assessment. Depression is therefore a separate
  **offline research baseline** (`live_compatible: false`). The loader refuses to run it on live input, and
  the registry already records that frontal-asymmetry screening needs two electrodes (F3/F4), which this
  single-channel kit cannot record. It is not a diagnostic tool.
- **Transfer is unvalidated.** Public data is multichannel microvolts from other amplifiers. Only
  amplitude-independent features are used (relative band powers, Hjorth mobility/complexity, spectral
  entropy; absolute variance is excluded), but differences in amplifier, electrode site and reference can
  still break transfer. Every live prediction carries that limitation text.
- Scores come from few subjects. The trainer reports the single held-out test split, a subject-level
  bootstrap interval and subject-wise cross-validation next to a majority-class baseline. Trust the interval
  and cross-validation more than the single number, and never compare against window-level (leaky) figures
  from papers.

## Setup (your computer)

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-eeg-data.txt
```

## Stress / mental load: PhysioNet "EEG During Mental Arithmetic Tasks"

```bash
mkdir -p data && cd data
wget -r -N -c -np https://physionet.org/files/eegmat/1.0.0/      # open access; check the page for license/citation
cd ..
.venv/bin/python train_public_eeg.py --dataset eegmat --root data/physionet.org/files/eegmat/1.0.0 \
    --channel Fp1 --output-dir eeg_model_stress
```

`SubjectNN_1.edf` is background EEG before the task (`rest`), `SubjectNN_2.edf` is EEG during arithmetic
(`mental_task`). Fp1 is the forehead site closest to the kit's placement. If the channel name differs, the
error lists the names in the file. Baseline recordings are usually longer than task recordings, so classes
are imbalanced; the trainer uses balanced class weights and balanced accuracy, and prints window counts.

## Depression (offline baseline): MDD versus healthy controls

Download the "EEG data: MDD patients and healthy controls" files from the figshare record by Mumtaz et al.
(search that title; read its license and cite the authors), unzip, then:

```bash
.venv/bin/python train_public_eeg.py --dataset mumtaz --root data/mumtaz --condition EC \
    --channel Fp1 --output-dir eeg_model_depression_offline
```

File names are expected like `MDD S1 EC.edf` / `H S3 EO.edf` (group, subject, condition). If yours differ,
the error prints the names it saw. MODMA needs an application form and a different layout: convert its
recordings to one-dimensional `.npy` files and use `--dataset manifest` (below).

## Any other dataset (SAM 40, DEAP, STEW, MODMA, ...)

Convert one channel per recording to a 1-D `.npy` file and write:

```json
{"recordings": [{"path": "s01_rest.npy", "subject": "s01", "label": "rest", "fs": 128},
                {"path": "s01_stress.npy", "subject": "s01", "label": "stress", "fs": 128}]}
```

```bash
.venv/bin/python train_public_eeg.py --dataset manifest --manifest my.json --task-name "rest vs stress" \
    --output-dir eeg_model_custom
```

Manifest runs are marked offline (`live_compatible: false`) until you review the result and change it
deliberately. Check each dataset's license and label definition first.

## What the trainer guarantees

Whole subjects are assigned to train/validation/test before windowing (stratified by label group); windows
do not overlap (6 s: 4 s analysed plus 1 s filter guards each side); windows with non-finite values,
flatlines or extreme outliers are rejected and counted; the model type is chosen on validation balanced
accuracy only and the test split is scored once; duplicate recordings and existing output folders are
refused. Outputs: `model.joblib`, `metadata.json`, `evaluation.json`, `manifest.json` (subject split and
recording hashes), `confusion_matrix.svg`. Candidates never replace anything automatically.

## Using a stress artifact in the dashboard

```bash
export BHM_EEG_MODEL_DIR=eeg_model_stress
.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Use the same Python and scikit-learn versions that trained it (checked at load). `/api/eeg/models/registry`
then marks only `stress_state_index` available, and `/eeg` shows the label with the unvalidated-transfer
limitation. Simulated signals never reach the classifier.

## Reading the results honestly

A result near the majority baseline means the features cannot separate the classes. A high single-split
score with a wide bootstrap interval is weak evidence. Public-data performance does not predict kit
performance: validate on your own recordings (see the protocol) before saying anything about the device.
