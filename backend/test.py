import torch
import re
import sounddevice as sd
import numpy as np
import queue
import soundfile as sf
import threading
import time
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from collections import Counter
from pyannote.audio import Pipeline

# Load model dan processor
model_name = "Gaben69181/wav2vec2-large-xlsr-id-colab-fine-tuning"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

# Load dictionary kata dasar
def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(open('katadasar.txt').read()))

def P(word, N=sum(WORDS.values())):
    return WORDS[word] / N

def correction(word):
    return max(candidates(word), key=P)

def candidates(word):
    return known([word]) or known(edits1(word)) or known(edits2(word)) or [word]

def known(words):
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits  = [(word[:i], word[i:]) for i in range(len(word)+1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_sentence(sentence):
    return ' '.join([correction(w) for w in sentence.split()])

# Audio settings
samplerate = 16000
duration = 5  # detik
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def get_transcription_segments(audio_input, pipeline):
    # Simpan audio batch ke file sementara
    temp_wav = "temp_batch.wav"
    sf.write(temp_wav, audio_input, samplerate)

    # Jalankan diarization
    diarization = pipeline(temp_wav)
    # audio_len = len(audio_input) / samplerate # Not strictly needed for output

    # Baca ulang audio untuk slicing per segmen
    audio_data, _ = sf.read(temp_wav)

    processed_segments_keys = set()
    output_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # Avoid double processing by using unique (start, end, speaker) as key
        segment_key = (turn.start, turn.end, speaker)
        if segment_key in processed_segments_keys:
            continue
        processed_segments_keys.add(segment_key)

        start_sample = int(turn.start * samplerate)
        end_sample = int(turn.end * samplerate)
        segment_audio = audio_data[start_sample:end_sample]

        if len(segment_audio) == 0:
            continue

        # Transkripsi hanya segmen ini
        input_values = processor(segment_audio, return_tensors="pt", sampling_rate=samplerate).input_values
        with torch.no_grad():
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        cleaned = clean_text(transcription)
        corrected = correct_sentence(cleaned)
        
        output_segments.append({
            'start_rel': turn.start,
            'end_rel': turn.end,
            'speaker': speaker,
            'text': corrected
        })
    return output_segments


def record_audio(pipeline):
    chunk_list = []
    processed_until_time = 0.0
    chunk_duration = duration # Use global duration as chunk_duration

    with sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback):
        print("🎙️ Mulai rekam (Ctrl+C untuk berhenti)...")
        try:
            while True:
                # 1. Record a new chunk
                audio_buffer = []
                start_time_chunk_rec = time.time()
                while time.time() - start_time_chunk_rec < chunk_duration:
                    audio_chunk_data = q.get()
                    if audio_chunk_data.ndim == 2:
                        audio_buffer.append(audio_chunk_data[:, 0])
                    else:
                        audio_buffer.append(audio_chunk_data)
                new_chunk = np.concatenate(audio_buffer)
                
                # 2. Add new_chunk to chunk_list
                chunk_list.append(new_chunk)

                # 3. If not enough chunks, continue
                if len(chunk_list) < 2:
                    continue

                # 4. We have at least two chunks. Process the first two.
                chunk_A = chunk_list[0]
                chunk_B = chunk_list[1]

                # 5. Combine chunk_A and chunk_B
                combined_audio = np.concatenate((chunk_A, chunk_B))
                
                # 6. Get transcription segments for the combined audio
                #    Timestamps in 'segments' are relative to the start of combined_audio
                segments = get_transcription_segments(combined_audio, pipeline)

                # 7. Iterate through segments and print those starting in chunk_A
                #    Absolute start time of chunk_A is current_processing_start_time
                current_processing_start_time_for_chunk_A = processed_until_time
                
                for seg in segments:
                    # We only finalize and print segments that start within chunk_A's duration
                    if seg['start_rel'] < chunk_duration:
                        abs_start = current_processing_start_time_for_chunk_A + seg['start_rel']
                        abs_end = current_processing_start_time_for_chunk_A + seg['end_rel']
                        
                        # To avoid re-printing parts of segments that were already covered by the end of a previous chunk_A's segment.
                        # This check ensures we only print new information.
                        if abs_start >= processed_until_time:
                             print(f"[{abs_start:.1f}s - {abs_end:.1f}s] {seg['speaker']}: {seg['text']}\n")

                # 8. Update processed_until_time
                processed_until_time += chunk_duration
                
                # 9. Remove the processed chunk_A from the list
                chunk_list.pop(0)

        except KeyboardInterrupt:
            print("\n🛑 Dihentikan.")

# === MAIN ===
if __name__ == "__main__":
    # Load diarization pipeline
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        HF_TOKEN = "YOUR_HF_TOKEN" # input("Masukkan Hugging Face token Anda: ").strip()

    print("🔁 Memuat pipeline PyAnnote...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)

    # Mulai proses realtime
    record_audio(pipeline)
