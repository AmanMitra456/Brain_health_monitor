# Recording protocol for the Maker UNO + BioAmp kit

A starting protocol for collecting labeled single-channel recordings, so models can be trained and
validated on the real hardware. It is a proposal for the investigators to review, not a validated
procedure. Electrode placement, referencing, skin preparation and safety are defined in
[HARDWARE.md](HARDWARE.md); follow that document and keep placement identical for every session.

## Before collecting

- Written informed consent covering storage of the recordings and their research use. Recordings
  contain no identity fields; keep any subject key separately and use codes (`s01`, `s02`, ...).
- Aim for as many independent participants as you can recruit (the more the better; fewer than about
  20 cannot support general claims) and several sessions each. All recordings of one participant stay in
  one split.
- Record in a quiet room, 50 Hz mains filter setting matching your country, one backend worker, and
  close other serial programs.

## Conditions (one recording per condition per session)

| Label | What the participant does | Length |
|---|---|---|
| `rest` | Eyes open, looking at a fixed point, relaxed, minimal blinking | 3 min |
| `focus` | Sustained-attention task without time pressure, e.g. silently counting backwards in 7s from a given number | 3 min |
| `stress` | Same kind of task under time pressure with a visible countdown and a penalty for errors (for example timed mental arithmetic or a Stroop task) | 3 min |

- Randomize condition order across participants and rest 2 min between conditions to reduce order effects.
- Ask for a self-report rating at the end of each condition (for example 1-9 for stress and for
  concentration) and store it in the manifest `notes`. Labels come from the protocol, but ratings show
  whether the condition really worked for that person.
- Avoid speaking, chewing, large head movements and deliberate blinks; muscle and eye artifacts are the
  main confounds on a forehead channel and can make a classifier "succeed" for the wrong reason.

## Recording and converting

1. Open `/eeg`, choose the real serial port (never the simulated source), connect, wait for the quality
   state to read Acceptable, then **Start recording**; stop at the end of the condition.
2. Export the CSV from the recordings list.
3. Convert and label it (refuses simulated data and bad clocks):

```bash
.venv/bin/python prepare_eeg_recordings.py --export eeg-<id>.csv --subject s01 --label focus \
    --out-dir my_eeg_data --manifest my_eeg_data/manifest.json
```

4. Edit `my_eeg_data/manifest.json`: fill in `task`, `provenance`, `license`, and set every recording's
   `split` to `train`, `validation` or `test`, giving each **subject** exactly one split. The trainer
   rejects `unassigned`, repeated subjects across splits and duplicate files.
5. Train on real hardware data (raw ADC counts, existing trainer):

```bash
.venv/bin/python train_eeg.py --manifest my_eeg_data/manifest.json --output-dir eeg_model_kit_candidate
```

## Using public data as a head start

Train on public data first ([EEG_PUBLIC_DATA.md](EEG_PUBLIC_DATA.md)), then check that model on the
held-out kit participants. If it performs near the majority baseline there, rely on the kit-trained model
instead. Do not mix public and kit recordings in one training file without recording that decision.

## Limits to state in any report

One channel and uncalibrated 10-bit counts; no clinical validation; stress and focus labels come from
tasks, not from a diagnosis; and nothing here assesses depression or any other disorder.
