import sys
import queue
import threading
import numpy as np
import torch
import sounddevice as sd
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.cluster import AgglomerativeClustering
import torch.nn.functional as F

# Model & Processor
model_name = "Gaben69181/wav2vec2-large-xlsr-id-colab-fine-tuning"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SAMPLE_RATE = 16000
SEGMENT_SECONDS = 2

class RealTimeDiarization(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Speaker Diarization + STT (ID)")
        self.resize(800, 600)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.audio_queue = queue.Queue()
        self.embeddings = []
        self.labels = []

        self.start_audio_stream()

        self.worker_thread = threading.Thread(target=self.process_audio_loop)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def start_audio_stream(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=int(SAMPLE_RATE * SEGMENT_SECONDS),
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        audio_data = indata[:, 0].copy()  # mono audio
        self.audio_queue.put(audio_data)

    def get_embedding(self, waveform_np):
        input_values = processor(waveform_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            hidden_states = model.wav2vec2(input_values).last_hidden_state  # (1, seq_len, hidden_dim)
        embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def transcribe(self, waveform_np):
        input_values = processor(waveform_np, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(pred_ids)[0].lower()
        return transcription

    def process_audio_loop(self):
        while True:
            audio_data = self.audio_queue.get()
            if len(audio_data) < SAMPLE_RATE * SEGMENT_SECONDS:
                # pad if last segment is shorter
                padding = np.zeros(int(SAMPLE_RATE * SEGMENT_SECONDS) - len(audio_data), dtype=np.float32)
                audio_data = np.concatenate([audio_data, padding])

            embedding = self.get_embedding(audio_data)
            self.embeddings.append(embedding)

            # Clustering incremental (simple approach: cluster all embeddings so far)
            if len(self.embeddings) > 1:
                clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
                self.labels = clustering.fit_predict(np.array(self.embeddings))
            else:
                self.labels = [0]

            transcription = self.transcribe(audio_data)

            speaker_id = self.labels[-1]
            display_text = f"[Speaker {speaker_id}] {transcription}\n"

            # Update GUI safely from thread
            self.text_edit.append(display_text)

def main():
    app = QApplication(sys.argv)
    window = RealTimeDiarization()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
