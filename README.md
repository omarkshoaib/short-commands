## Arabic Command Transcription Comparator (Wav2Vec2 + Whisper Turbo)

This app records short Arabic commands on a button press, then transcribes with two engines and logs results to CSV:
- Wav2Vec2 (HF Arabic)
- Whisper large-v3-turbo (faster-whisper)

### Prereqs (Ubuntu/Debian)
- Python 3.10+
- PortAudio for `sounddevice`:
  - `sudo apt-get install python3-tk portaudio19-dev`

### Install
```
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### (Optional) MFCC-CNN front-end
You can train and enable a small MFCC-CNN to classify short Arabic commands quickly and, if confident, skip heavy ASR.

### Run
```
python app.py
```
- Hold the button to record; release to transcribe.
- CSV: `results/transcriptions.csv` with columns:
  - `timestamp, audio_file, audio_duration_ms, mfcc_label, mfcc_confidence, mfcc_used, wav2vec2, w2v2_confidence, wav2vec2_mapped, whisper_turbo, whisper_used, whisper_mapped, wav2vec2_time_ms, whisper_time_ms, total_processing_time_ms`

### Configure models (env or UI)
- `W2V2_MODEL_ID` (default `elgeish/wav2vec2-large-xlsr-53-arabic`)
- `WHISPER_MODEL` (default `large-v3-turbo`)
- `STT_DEVICE` (`cpu` or `cuda`, default `cpu`)
- `WHISPER_COMPUTE_TYPE` (default `int8`)

Notes
- First runs may download HF/faster-whisper weights.
- Ensure `VOSK_MODEL_PATH` points to the unzipped directory.

## MFCC-CNN training (branch: training-a-mfcc-cnn-model)

Quick start:
```
. .venv/bin/activate
python training/train_mfcc_cnn.py --config training/config.yaml
python training/eval_mfcc_cnn.py  --config training/config.yaml
```
- Data expected under `data/` with CSVs `train.csv`, `val.csv`, `test.csv` and wavs under `data/dataset/dataset/<class>/*.wav`.
- Background noise (optional aug): `data/background_noise/background_noise/*.wav`.
- Checkpoints: `checkpoints/mfcc_cnn/best.pt`, label map: `exports/mfcc_cnn/label_map.json`.

Config knobs (`training/config.yaml`): sample rate, MFCC params, augment SNR/time shift/gain, batch size, epochs, etc.

