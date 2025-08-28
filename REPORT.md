# Arabic Command Comparator – System Report

This document explains the full workflow, components, policies, and tuning knobs of the Arabic short-command system. It covers both GUI and CLI runners.

## High-level pipeline

```mermaid
graph TD
  A["Mic input<br/>16 kHz mono"] --> B["PortAudio<br/>RawInputStream"]
  B --> C["VAD + Ring Buffer<br/>(webrtcvad)<br/>Pre-roll buffered<br/>Min speech gate"]
  C --> D{KWS enabled?}
  D -- No --> E["Primary STT<br/>Wav2Vec2 (HF)"]
  D -- Yes --> F["Temporal KWS<br/>(CMVN MFCC + DTW)<br/>Dynamic thresholds"]
  F -- Pass --> E
  F -- Reject --> X["Reject utterance<br/>Append CSV/telemetry"]
  E --> G["Command mapping<br/>(Arabic norm + Levenshtein)"]
  E --> H{Fallback policy?}
  H -- Yes --> I["Fallback STT<br/>Whisper large-v3-turbo"]
  H -- No --> J[Skip Whisper]
  I --> K[Command mapping]
  G --> L[CSV + Telemetry JSONL]
  K --> L
```

## Components

- Audio capture: `sounddevice.RawInputStream` at 16 kHz, mono, int16.
- VAD + ring buffer: `webrtcvad` processes fixed-size frames, keeps pre-roll (ring buffer), gates too-short utterances, writes a VAD-trimmed temp WAV.
- KWS (wake-word):
  - Templates: CMVN-normalized MFCCs from `kws/templates/<profile>/*.wav`.
  - Scoring: subsequence DTW with Sakoe–Chiba band (cosine frame distance), score mapped to [0,1] via 1/(1+d).
  - Temporal/dynamic gating: primary high/low thresholds; optional secondary checker with stricter band.
- Primary STT: Wav2Vec2 (Hugging Face), Arabic model (default `elgeish/wav2vec2-large-xlsr-53-arabic`). Confidence = mean of max softmax over frames.
- Fallback STT: faster-whisper (`large-v3-turbo`), device/compute-type per config (CPU int8 by default).
- Command mapping: Arabic normalization + Levenshtein snapping to `config.COMMANDS`, with a max edit distance cap.
- Logging: enforced-schema CSV and JSONL telemetry; offline ROC-like script for KWS threshold tuning.

## CLI run sequence

```mermaid
sequenceDiagram
  participant U as User (keyboard)
  participant CLI as CLI runner
  participant Rec as Recorder
  participant VAD as VAD/Ring
  participant KWS as Temporal KWS
  participant W2V2 as Wav2Vec2
  participant WH as Whisper
  participant CSV as CSV/Telemetry

  U->>CLI: r (start)
  CLI->>Rec: start()
  U->>CLI: s (stop)
  CLI->>Rec: stop (stream only)
  CLI->>VAD: record_with_vad() → vad_trim.wav
  alt KWS enabled
    CLI->>KWS: temporal_gate(vad_trim.wav)
    KWS-->>CLI: passed, score
    opt reject
      CLI->>CSV: write KWS-only row
      CLI-->>U: print "KWS rejected"
    end
  end
  CLI->>W2V2: transcribe_with_confidence()
  W2V2-->>CLI: text, conf
  CLI->>CLI: map_to_commands()
  CLI->>WH: transcribe() [if fallback triggered]
  WH-->>CLI: text
  CLI->>CLI: map_to_commands()
  CLI->>CSV: append full row
  CLI-->>U: print summary lines
```

## KWS details (temporal + dynamic)

- Templates per profile: `kws/templates/<profile>/*.wav` (enrolled via GUI checkbox or by dropping WAVs).
- Scoring: CMVN MFCC → subsequence DTW with band constraint; min distance → score = 1/(1+d).
- Temporal/dynamic thresholds:
  - Primary pass if `score ≥ KWS_THR_HIGH` or `score ≥ KWS_THR_LOW`.
  - Optional secondary checker with narrower band must also pass.
- SNR-aware thresholding is supported in KWS classic gate (optional), and dynamic thresholds concept aligns with Apple’s approach.

```mermaid
graph LR
  A[Input CMVN MFCC] --> S["Subsequence DTW<br/>band=KWS_BAND_FRAC"]
  T[Profile Templates] --> S
  S --> P["score ≥ THR_HIGH<br/>or ≥ THR_LOW"]
  P -- No --> R[Reject]
  P -- Yes --> Q{Require secondary?}
  Q -- No --> G[Gate open]
  Q -- Yes --> S2["Subsequence DTW<br/>band=KWS_SECONDARY_BAND"]
  S2 --> G2{score ≥ THR_LOW?}
  G2 -- Yes --> G
  G2 -- No --> R
```

## Fallback policy (mapping-aware)

- Whisper runs when any of the following is true:
  - Duration ≥ `WHISPER_FALLBACK_LONG_MS`.
  - W2V2 confidence < `W2V2_CONF_THRESHOLD` (stricter if SNR < `SNR_LOW_DB`).
  - W2V2 text empty.
  - W2V2 text failed command mapping and normalized distance > 0.35.

```mermaid
graph TD
  A["W2V2 text + conf"] --> B{"Long audio"}
  A --> C{"Low conf (SNR-aware)"}
  A --> D{"Empty or mapping bad"}
  B -- Yes --> E["Run Whisper"]
  C -- Yes --> E
  D -- Yes --> E
  B -- No --> F["Skip Whisper"]
  C -- No --> F
  D -- No --> F
```

## Command mapping

- Arabic normalization: remove diacritics and tatweel, normalize (إ/أ/آ→ا, ى→ي, ة→ه), collapse spaces.
- Levenshtein distance to each candidate in `config.COMMANDS`; select best within `CMD_MAP_MAX_DISTANCE`.
- Mapped outputs are written as `wav2vec2_mapped` / `whisper_mapped`.

## Data outputs

- CSV (schema enforced, auto-rotate legacy):
  - `timestamp, audio_file, audio_duration_ms, kws_passed, kws_score, wav2vec2, w2v2_confidence, wav2vec2_mapped, whisper_turbo, whisper_used, whisper_mapped, wav2vec2_time_ms, whisper_time_ms, total_processing_time_ms`.
- Telemetry JSONL (`results/telemetry.jsonl`): KWS/ASR details, duration, SNR per sample.
- Offline ROC-like script: `analysis_roc.py` sweeps KWS thresholds and prints precision/recall/FPR to guide choosing `KWS_THR_HIGH/LOW`.

## Terminology (quick)

- MFCC (Mel-Frequency Cepstral Coefficients): compact spectral features computed per short frame (e.g., 25 ms) that capture the phonetic shape of speech; standard for KWS/ASR front-ends.
- CMVN (Cepstral Mean and Variance Normalization): per-utterance normalization making each MFCC dimension zero-mean, unit-variance; reduces channel and loudness effects, improving template matching.
- Thresholds:
  - KWS_THR_HIGH / KWS_THR_LOW: dynamic wake thresholds; pass if score ≥ HIGH (confident) or ≥ LOW (borderline). Tune via telemetry/ROC.
  - KWS_BAND_FRAC / KWS_SECONDARY_BAND_FRAC: DTW band widths; lower is stricter (secondary checker is narrower).
  - W2V2_CONF_THRESHOLD: minimum Wav2Vec2 confidence; below triggers Whisper (especially if SNR is poor).
  - WHISPER_FALLBACK_LONG_MS: duration above which Whisper is preferred for final accuracy.
  - CMD_MAP_MAX_DISTANCE: max Levenshtein distance to snap to a known command.

## Configuration (env / `config.py`)

- Audio: `SAMPLE_RATE, VAD_AGGRESSIVENESS, VAD_FRAME_MS, RING_BUFFER_MS, MIN_SPEECH_MS`.
- KWS: `KWS_ENABLED, KWS_TEMPLATES_DIR, KWS_PROFILE_ID, KWS_THR_HIGH/LOW, KWS_BAND_FRAC, KWS_SECONDARY_BAND_FRAC`.
- STT:
  - W2V2: `W2V2_MODEL_ID, STT_DEVICE`.
  - Whisper: `WHISPER_MODEL, WHISPER_COMPUTE_TYPE`.
- Fallback policy: `WHISPER_FALLBACK_LONG_MS, W2V2_CONF_THRESHOLD, SNR_LOW_DB, ALWAYS_RUN_WHISPER`.
- Commands: `COMMANDS, CMD_MAP_MAX_DISTANCE`.
- Logging: `RESULTS_CSV, TELEMETRY_PATH`.

## CLI vs GUI

- GUI (Tkinter): press-and-hold button, enrollment checkbox, profile field; results pane mirrors CSV fields.
- CLI: `python cli.py` → r=start, s=stop, q=quit; prints KWS/ASR summaries and appends CSV/telemetry.

## Tuning guidance

- KWS: use ROC script to balance FP (false wakes) vs FN (misses). Start with `KWS_THR_HIGH≈0.70`, `KWS_THR_LOW≈0.60`, then refine with more data.
- Fallback: if W2V2 is often slightly off, lower the mapping cutoff or enrich `COMMANDS` with variants. If latency is higher than desired, raise `WHISPER_FALLBACK_LONG_MS`.
- VAD: increase `VAD_AGGRESSIVENESS` in noisy rooms; adjust `MIN_SPEECH_MS` to reduce spurious short clips.

## References

- Apple Hey Siri: dynamic thresholds and temporal integration.
  - Hey Siri: https://machinelearning.apple.com/research/hey-siri
  - Voice Trigger: https://machinelearning.apple.com/research/voice-trigger
- Real-time vs turn-based voice agents: streaming design choices.
  - Softcery guide: https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture/

