## Arabic Command Transcription Comparator (Wav2Vec2 + Whisper Turbo + Vosk)

This app records short Arabic commands on a button press, then transcribes with three engines and logs results to CSV:
- Wav2Vec2 (HF Arabic)
- Whisper large-v3-turbo (faster-whisper)
- Vosk (offline Arabic model)

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

### Vosk Arabic model
Download and unzip one model into `models/` (path configurable via `VOSK_MODEL_PATH`):
- Linto 1.1.0: `https://alphacephei.com/vosk/models/vosk-model-ar-0.22-linto-1.1.0.zip`
- MGB2 0.4: `https://alphacephei.com/vosk/models/vosk-model-ar-mgb2-0.4.zip`

### Run
```
python app.py
```
- Hold the button to record; release to transcribe.
- CSV: `results/transcriptions.csv` with columns:
  - `timestamp, audio_file, audio_duration_ms, wav2vec2, whisper_turbo, vosk, wav2vec2_time_ms, whisper_time_ms, vosk_time_ms, total_processing_time_ms`

### Configure models (env or UI)
- `W2V2_MODEL_ID` (default `elgeish/wav2vec2-large-xlsr-53-arabic`)
- `WHISPER_MODEL` (default `large-v3-turbo`)
- `VOSK_MODEL_PATH` (default `models/vosk-model-ar-0.22-linto-1.1.0`)
- `STT_DEVICE` (`cpu` or `cuda`, default `cpu`)
- `WHISPER_COMPUTE_TYPE` (default `int8`)

Notes
- First runs may download HF/faster-whisper weights.
- Ensure `VOSK_MODEL_PATH` points to the unzipped directory.

