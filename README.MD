# STT-Diarization-App

A FastAPI-based backend for real-time Speech-to-Text (STT) and Speaker Diarization using Hugging Face Transformers, pyannote.audio, and Silero VAD.  
Supports streaming audio from the browser, speaker separation, and transcription in Indonesian.

---

## Features

- **Real-time Speech-to-Text (STT)** using Wav2Vec2 models
- **Speaker Diarization** (who spoke when) using pyannote.audio
- **Voice Activity Detection (VAD)** with Silero VAD for efficient speech segmentation
- **WebSocket API** for low-latency audio streaming from browser or client
- **Frontend static file serving** (React app or other SPA)
- **Easy deployment** with Uvicorn

---

## Folder Structure

```
stt-diarization-app/
├── backend/
│   ├── main.py           # FastAPI backend (WebSocket, STT, diarization, VAD)
│   ├── test.py           # Example/test script for local audio processing
│   ├── requirements.txt  # Python dependencies
│   └── ...               # Other backend files
├── frontend_dist/
│   └── ...               # Built frontend assets (served by backend)
└── README.md             # This file
```

---

## Requirements

- Python 3.8+
- FFmpeg (must be in PATH, for audio decoding)
- [Hugging Face account](https://huggingface.co/) and access tokens for some models
- (Optional) CUDA GPU for faster inference

### Python Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

**Key packages:**
- fastapi
- uvicorn
- torch
- torchaudio
- transformers
- pyannote.audio
- pydub
- soundfile
- librosa
- webrtcvad
- python-dotenv

---

## Setup

### 1. Environment Variables

Create a `.env` file in `backend/` with your Hugging Face token:

```
HF_TOKEN=hf_xxx_your_huggingface_token
```

Or set it in your environment.

### 2. Download Model Weights

The backend will automatically download models on first run.  
You may need to accept model terms on Hugging Face for `pyannote/speaker-diarization`.

### 3. FFmpeg

Make sure `ffmpeg` is installed and available in your system PATH.  
[Download FFmpeg for Windows](https://www.gyan.dev/ffmpeg/builds/)

---

## Running the Backend

From the `backend/` directory:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

---

## WebSocket API

- **Endpoint:** `ws://localhost:8000/ws`
- **Send:** Raw audio bytes (e.g., WebM/Opus from browser)
- **Receive:** JSON messages with speaker label, transcription, and timestamps

### Example Message

```json
{
  "speaker": "SPEAKER_00",
  "text": "halo semuanya apa kabar",
  "start_time": 0.33,
  "end_time": 1.99
}
```

---

## Frontend

- Place your built frontend (e.g., React) in `frontend_dist/`
- Static files are served at `/assets`
- Main page is served at `/`

---

## Testing

You can use `test.py` for local batch audio testing and diarization.

---

## Troubleshooting

- **Model loading errors:** Check your Hugging Face token and internet connection.
- **Audio decoding errors:** Ensure FFmpeg is installed and in PATH.
- **Connection refused:** Make sure backend is running before connecting clients.
- **Only first few seconds transcribed:** Make sure client sends audio in correct format and signals end of utterance if needed.

---

## Credits

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Silero VAD](https://github.com/snakers4/silero-vad)

---

## License

MIT (or specify your license here)