from dotenv import load_dotenv # For .env file
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import logging
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pyannote.audio import Pipeline
import io
import soundfile as sf
import numpy as np
import os # For Hugging Face token
import librosa # Import librosa
import tempfile # For temporary file
from pydub import AudioSegment # Import pydub
import webrtcvad # Import webrtcvad

load_dotenv() # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Model Loading ---
# Ensure you have a Hugging Face token set as an environment variable or logged in via huggingface-cli
# For pyannote.audio, you might need to accept user agreements on the Hugging Face model pages.
HF_TOKEN = os.environ.get("HF_TOKEN") # Or load it from a .env file, or ensure user is logged in

# Initialize models to None
stt_model = None
stt_processor = None
diarization_pipeline = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    logger.info("Loading STT model and processor...")
    # Using the model name you updated to:
    stt_model_name = "indonesian-nlp/wav2vec2-large-xlsr-indonesian"
    # Prefer using HF_TOKEN environment variable if set, otherwise fallback or use hardcoded if necessary
    # For consistency with pyannote, let's assume HF_TOKEN is the primary method.
    # If you've hardcoded it because HF_TOKEN wasn't working, ensure it's the correct active token.
    # For this diff, I'll use HF_TOKEN, adjust if your setup requires the hardcoded one.
    token_to_use_stt = HF_TOKEN if HF_TOKEN else "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" # Fallback to your hardcoded one if HF_TOKEN is not set
    
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name, token=token_to_use_stt)
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name, token=token_to_use_stt)
    stt_model.eval() # Set model to evaluation mode
    stt_model.to(DEVICE)
    logger.info("STT model loaded successfully.")
except Exception as e_stt:
    logger.error(f"Error loading STT model: {e_stt}")
    # stt_model and stt_processor remain None

try:
    logger.info("Loading Speaker Diarization pipeline...")
    # You might need to agree to terms on Hugging Face for these models:
    # - pyannote/speaker-diarization-pytorch
    # - pyannote/segmentation-pytorch
    # Using use_auth_token=True if you've logged in via CLI and accepted terms,
    # or HF_TOKEN if you prefer to pass it explicitly and it's correctly set.
    # The error message suggested use_auth_token=YOUR_AUTH_TOKEN, so HF_TOKEN is appropriate.
    token_to_use_diar = HF_TOKEN if HF_TOKEN else True # Fallback to True if HF_TOKEN not set, assuming CLI login

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=token_to_use_diar
    )
    logger.info("Speaker Diarization pipeline loaded successfully.")
except Exception as e_diar:
    logger.error(f"Error loading Speaker Diarization pipeline: {e_diar}")
    # diarization_pipeline remains None

if stt_model and diarization_pipeline:
    logger.info(f"All models loaded. Using device: {DEVICE}")
elif stt_model:
    logger.warning(f"STT model loaded, but Diarization pipeline FAILED. Using device: {DEVICE}")
elif diarization_pipeline:
    logger.warning(f"Diarization pipeline loaded, but STT model FAILED. Using device: {DEVICE}")
else:
    logger.error(f"Both STT and Diarization models FAILED to load. Check logs above.")
# --- End Model Loading ---


# Serve frontend static files
# The path "/app/frontend_dist" corresponds to where files are copied in Dockerfile
# For local development, you might need to adjust this path or run frontend dev server separately.
# When running with Docker, this will serve the built React app.
app.mount("/assets", StaticFiles(directory="../frontend_dist/assets"), name="assets")

@app.get("/")
async def serve_index():
    return FileResponse("../frontend_dist/index.html")

@app.get("/{catchall:path}")
async def serve_react_routes(catchall: str):
    # This helps with client-side routing in React if you have routes like /about, /contact
    # It ensures that any path not matched by other API routes serves the index.html
    # allowing React Router to take over.
    # Check if the path looks like a file extension, if so, it's likely a static file request
    # that was missed by /assets or other static mounts.
    # This is a simple check and might need refinement.
    if "." in catchall.split("/")[-1]: # e.g. "favicon.ico"
        # Potentially serve from frontend_dist if it's a known static file type not in /assets
        # For now, we'll just let it go to index.html or it might 404 if not handled by StaticFiles
        pass
    return FileResponse("../frontend_dist/index.html")
# Helper function to convert audio bytes to required format (e.g., 16kHz mono)
# Renaming for clarity as this now returns float32 numpy array
def preprocess_browser_audio_to_float32_pcm(audio_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
    """
    Converts raw audio bytes from browser (e.g., webm/opus) to a float32 NumPy array
    of 16kHz mono PCM samples. Returns empty array on failure.
    """
    logger.debug(f"Preprocessing browser audio (len: {len(audio_bytes)} bytes)")
    try:
        # Try pydub first (for common formats like webm/opus)
        # pydub uses ffmpeg, ensure it's installed and in PATH
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1)
        audio_segment = audio_segment.set_frame_rate(target_sample_rate)
        # Get samples as a NumPy array, normalized to float32 in [-1, 1]
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32767.0
        logger.debug(f"pydub successful. Output shape: {samples.shape}")
        return samples
    except Exception as e_pydub:
        logger.warning(f"pydub failed to process browser audio: {e_pydub}. Trying soundfile...")
        # Fallback to soundfile (good for WAV or raw PCM if browser sends that)
        try:
            # Ensure soundfile gets a clean BytesIO object for each attempt
            audio_input, original_sample_rate = sf.read(io.BytesIO(audio_bytes), dtype='float32', always_2d=False)
            if audio_input.ndim > 1: # Should be mono after pydub attempt, but as a fallback
                audio_input = np.mean(audio_input, axis=1)
            
            if original_sample_rate != target_sample_rate:
                logger.info(f"Resampling from {original_sample_rate} Hz to {target_sample_rate} Hz using librosa (after soundfile load)")
                # Ensure audio_input is float32 for librosa.resample
                audio_input = librosa.resample(audio_input.astype(np.float32), orig_sr=original_sample_rate, target_sr=target_sample_rate)
            logger.debug(f"soundfile successful. Output shape: {audio_input.shape}")
            return audio_input.astype(np.float32)
        except Exception as e_sf:
            logger.error(f"soundfile also failed: {e_sf}")
            logger.error(f"Preprocessing failed for audio chunk (first 64 bytes: {audio_bytes[:64].hex()}). Ensure FFmpeg is installed and in PATH for pydub.")
            return np.array([], dtype=np.float32)

from starlette.websockets import WebSocketState # Ensure this is imported

# Silero VAD model and utilities - loaded globally once
SILERO_VAD_MODEL = None
SILERO_VAD_UTILS = None
try:
    torch.set_num_threads(1)
    SILERO_VAD_MODEL, SILERO_VAD_UTILS = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    logger.info("Silero VAD model loaded successfully.")
except Exception as e_silero:
    logger.error(f"Failed to load Silero VAD model: {e_silero}. VAD functionality will be disabled.")
    # SILERO_VAD_MODEL will remain None

async def process_vad_speech_segment(audio_float32_pcm: np.ndarray, websocket: WebSocket, sample_rate: int = 16000):
    """
    Processes a chunk of float32 PCM audio data (identified by VAD as speech)
    for STT and diarization.
    """
    if audio_float32_pcm.size == 0:
        logger.warning("process_vad_speech_segment called with empty audio data.")
        return

    try:
        audio_tensor_for_models = torch.from_numpy(audio_float32_pcm).float()
        
        diar_input = {"waveform": audio_tensor_for_models.unsqueeze(0).to(DEVICE), "sample_rate": sample_rate}
        logger.info(f"Running diarization on VAD speech segment of {len(audio_float32_pcm)/sample_rate:.2f}s...")
        diarization = diarization_pipeline(diar_input)
        logger.info("Diarization complete on VAD speech segment.")

        if not diarization.get_timeline().support():
            logger.info("No speakers found by diarization in VAD speech segment.")
            # Optionally send a system message if desired, but VAD already implies speech
            # if websocket.application_state == WebSocketState.CONNECTED:
            #     await websocket.send_json({"speaker": "System", "text": "[No speakers in segment]"})
            return

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            turn_start_sample = int(turn.start * sample_rate)
            turn_end_sample = int(turn.end * sample_rate)
            
            # Slice from the original float32 pcm passed to this function
            turn_audio_float32_pcm_segment = audio_float32_pcm[turn_start_sample:turn_end_sample]

            if turn_audio_float32_pcm_segment.size == 0:
                logger.info(f"Skipping empty diarization turn for speaker {speaker_label}")
                continue
            
            input_values = stt_processor(turn_audio_float32_pcm_segment, return_tensors="pt", sampling_rate=sample_rate).input_values
            input_values = input_values.to(DEVICE)
            
            with torch.no_grad():
                logits = stt_model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription_text = stt_processor.batch_decode(predicted_ids)[0]
            
            logger.info(f"Transcription for {speaker_label}: {transcription_text}")
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "speaker": speaker_label,
                    "text": transcription_text,
                    "start_time": turn.start,
                    "end_time": turn.end
                })
    except Exception as e_process:
        logger.error(f"Error during VAD speech segment processing: {e_process}", exc_info=True)
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"speaker": "Error", "text": "Server processing error for speech segment."})
        except Exception: pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    if not all([stt_model, stt_processor, diarization_pipeline, SILERO_VAD_MODEL]):
        err_msg = "One or more models (STT, Diarization, or Silero VAD) not loaded."
        logger.error(f"{err_msg} Closing connection.")
        try: await websocket.send_json({"error": err_msg})
        except Exception: pass
        try: await websocket.close(code=1011)
        except Exception: pass
        return

    # Silero VAD utilities
    get_speech_ts_fn = SILERO_VAD_UTILS[0] if SILERO_VAD_UTILS else None
    if not get_speech_ts_fn:
        logger.error("Silero VAD utility 'get_speech_timestamps' not available. Cannot perform VAD.")
        try: await websocket.send_json({"error": "VAD utility not available."})
        except Exception: pass
        try: await websocket.close(code=1011)
        except Exception: pass
        return

    SAMPLE_RATE = 16000
    # Buffer for raw bytes from browser
    browser_audio_accumulator = bytearray()
    # How much browser audio (likely compressed) to accumulate before trying to convert with pydub
    BROWSER_ACCUMULATION_THRESHOLD = 16000 # Approx 1s of WebM/Opus
    # Silero VAD parameters (can be tuned)
    MIN_SPEECH_DURATION_MS = 250
    MIN_SILENCE_DURATION_MS = 300 # Shorter silence to be more responsive
    SPEECH_PAD_MS = 100 # Pad speech segments slightly
    # We will process whatever preprocess_browser_audio_to_float32_pcm gives us.

    try:
        while True:
            browser_chunk = await websocket.receive_bytes()
            browser_audio_accumulator.extend(browser_chunk)
            logger.debug(f"WS recv: {len(browser_chunk)} bytes, browser_accumulator: {len(browser_audio_accumulator)}")

            if len(browser_audio_accumulator) >= BROWSER_ACCUMULATION_THRESHOLD:
                logger.info(f"Threshold reached. Processing browser_accumulator (size: {len(browser_audio_accumulator)}) with Silero VAD.")
                
                current_audio_bytes = bytes(browser_audio_accumulator)
                browser_audio_accumulator.clear()
                
                audio_float32_pcm = preprocess_browser_audio_to_float32_pcm(current_audio_bytes)

                if audio_float32_pcm.size == 0:
                    logger.warning("Preprocessing browser audio chunk yielded no PCM data for VAD.")
                    continue
                
                audio_tensor_for_vad = torch.from_numpy(audio_float32_pcm).float()
                
                # Use Silero VAD to get speech timestamps
                speech_timestamps = get_speech_ts_fn(
                    audio_tensor_for_vad,
                    SILERO_VAD_MODEL,
                    sampling_rate=SAMPLE_RATE,
                    min_speech_duration_ms=MIN_SPEECH_DURATION_MS,
                    min_silence_duration_ms=MIN_SILENCE_DURATION_MS, # This helps segment based on silence
                    speech_pad_ms=SPEECH_PAD_MS
                )
                logger.info(f"Silero VAD detected speech timestamps for current chunk: {speech_timestamps}")

                if not speech_timestamps:
                    logger.info("No speech detected by Silero VAD in current chunk.")
                    # No need to send "[Silence/Noise]" for every non-speech chunk if it's continuous
                    continue

                for ts in speech_timestamps:
                    start_sample, end_sample = ts['start'], ts['end']
                    speech_segment_float32_pcm = audio_float32_pcm[start_sample:end_sample]
                    if speech_segment_float32_pcm.size > 0:
                        logger.info(f"Processing VAD-detected speech segment: {len(speech_segment_float32_pcm)/SAMPLE_RATE:.2f}s")
                        await process_vad_speech_segment(speech_segment_float32_pcm, websocket)
                    else:
                        logger.info("Skipping zero-size segment from Silero VAD timestamps.")


    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected. Final browser_accumulator size: {len(browser_audio_accumulator)} bytes.")
        if len(browser_audio_accumulator) > 0: # Process any remaining data
            logger.info("Processing final accumulated browser audio on disconnect.")
            audio_float32_pcm = preprocess_browser_audio_to_float32_pcm(bytes(browser_audio_accumulator))
            if audio_float32_pcm.size > 0:
                # For simplicity, process the whole remaining chunk as one segment
                # More advanced: could run VAD on this final chunk too.
                await process_vad_speech_segment(audio_float32_pcm, websocket)
        logger.info("Final processing on disconnect complete.")
            
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket main loop: {e}", exc_info=True)
    finally:
        logger.info("WebSocket endpoint finishing.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)