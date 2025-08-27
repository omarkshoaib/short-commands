## Arabic Command Transcription Comparator (Wav2Vec2 + Whisper Turbo)

This app records short Arabic commands on a button press, then transcribes with two engines and logs results to CSV:
- Wav2Vec2 (HF Arabic model)
- Whisper large-v3-turbo (via faster-whisper)

### Prereqs (Ubuntu/Debian)
- Python 3.10+
- PortAudio (for `sounddevice`):
  - `sudo apt-get install python3-tk portaudio19-dev`

### Install
```
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run
```
python app.py
```
- Press and hold the button to record. Release to transcribe.
- Results append to `results/transcriptions.csv` with columns: timestamp, audio_file, wav2vec2, whisper_turbo.

### Configure models (env or UI)
- `W2V2_MODEL_ID` (default `elgeish/wav2vec2-large-xlsr-53-arabic`)
- `WHISPER_MODEL` (default `large-v3-turbo`, supports `large-v3` as well)

Notes
- First run may download HF weights and/or faster-whisper models; cached afterward.
- If torch install fails, follow the PyTorch site for a platform-specific wheel.

