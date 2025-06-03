import { useState, useEffect, useRef } from 'react';
import './App.css';

interface TranscriptionSegment {
  speaker: string;
  text: string;
  start_time?: number; // Optional for now, but good to have
  end_time?: number;   // Optional for now, but good to have
}

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [transcription, setTranscription] = useState<TranscriptionSegment[]>([]);
  // Placeholder for audio visualizer data if needed
  // const [audioData, setAudioData] = useState<Uint8Array | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);


  const MOCK_SOCKET_URL = 'ws://localhost:8000/ws'; // Replace with your actual backend WebSocket URL

  useEffect(() => {
    // Cleanup WebSocket on component unmount
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, []);

  const drawVisualizer = () => {
    if (!analyserRef.current || !dataArrayRef.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext('2d');
    if (!canvasCtx) return;

    analyserRef.current.getByteTimeDomainData(dataArrayRef.current);

    canvasCtx.fillStyle = 'rgb(200, 200, 200)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 0, 0)';
    canvasCtx.beginPath();

    const sliceWidth = (canvas.width * 1.0) / analyserRef.current.frequencyBinCount;
    let x = 0;

    for (let i = 0; i < analyserRef.current.frequencyBinCount; i++) {
      const v = dataArrayRef.current[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();

    animationFrameIdRef.current = requestAnimationFrame(drawVisualizer);
  };


  const setupAudioProcessing = async (stream: MediaStream) => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    
    const source = audioContextRef.current.createMediaStreamSource(stream);
    analyserRef.current = audioContextRef.current.createAnalyser();
    analyserRef.current.fftSize = 2048;
    const bufferLength = analyserRef.current.frequencyBinCount;
    dataArrayRef.current = new Uint8Array(bufferLength);

    source.connect(analyserRef.current);
    // We don't connect analyser to destination to avoid feedback, unless you want to hear the mic input
    // analyserRef.current.connect(audioContextRef.current.destination);

    // Start visualizer
    if (canvasRef.current) {
        drawVisualizer();
    }

    // Setup WebSocket connection
    socketRef.current = new WebSocket(MOCK_SOCKET_URL);

    socketRef.current.onopen = () => {
      console.log('WebSocket connection established');
      // Send audio data in chunks
      // This part needs a proper audio chunking mechanism
      // For simplicity, we'll simulate sending some data or rely on a more robust library
    };

    socketRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data as string);
        logger.info("Received from server:", data); // Log received data

        if (data.error) {
          console.error('Error from server:', data.error);
          setTranscription((prev) => [...prev, { speaker: "Error", text: data.error }]);
          return;
        }

        if (data.speaker && data.text) {
          setTranscription((prev) => [...prev, {
            speaker: data.speaker,
            text: data.text,
            start_time: data.start_time,
            end_time: data.end_time
          }]);
        } else if (data.text) {
          // This case might be for interim, non-diarized results or system messages
          // For now, we'll treat it as a generic message from a "System" speaker
          // if no speaker is explicitly provided.
          setTranscription((prev) => [...prev, { speaker: data.speaker || "System", text: data.text }]);
        } else {
          // Handle other types of messages if necessary
          logger.warn("Received unknown message format from server:", data);
        }
      } catch (error) {
        console.error('Error processing message from server:', event.data, error);
        setTranscription((prev) => [...prev, { speaker: "System", text: `Error processing server message: ${event.data}` }]);
      }
    };

    socketRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setTranscription((prev) => [...prev, { speaker: "Error", text: "WebSocket connection error." }]);
    };

    socketRef.current.onclose = () => {
      console.log('WebSocket connection closed');
      setTranscription((prev) => [...prev, { speaker: "System", text: "Disconnected from server." }]);
    };
    
    // Example of sending audio data (needs proper implementation)
    // This is a very simplified way and might not work as expected for real-time streaming.
    // You'd typically use a ScriptProcessorNode (deprecated) or AudioWorkletNode for this.
    
    // Attempt to set a mimeType. Browsers have varying support.
    // Common options: 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/wav'
    // 'audio/wav' would be ideal for simplicity on backend if supported, but often isn't for MediaRecorder.
    // Let's try common ones and fall back.
    const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/ogg;codecs=opus',
        'audio/wav', // Less likely to be supported for recording by all browsers
        // Add more or leave empty to use browser default
    ];
    let selectedMimeType = '';
    for (const mimeType of mimeTypes) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
            selectedMimeType = mimeType;
            break;
        }
    }
    logger.info(`Using mimeType: ${selectedMimeType || 'browser default'}`);

    const mediaRecorder = selectedMimeType
        ? new MediaRecorder(stream, { mimeType: selectedMimeType })
        : new MediaRecorder(stream);
        
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
            socketRef.current.send(event.data);
        }
    };
    mediaRecorder.start(1000); // Send data every 1 second, adjust as needed

  };


  const handleToggleRecording = async () => {
    if (isRecording) {
      // Stop recording
      setIsRecording(false);
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        mediaStreamRef.current = null;
      }
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
        animationFrameIdRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
         // audioContextRef.current.close(); // Closing context here might be too soon if you want to restart
         // audioContextRef.current = null;
      }
       // Clear visualizer
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');
        if (canvasCtx) {
            canvasCtx.fillStyle = 'rgb(200, 200, 200)';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
        }
      }

    } else {
      // Start recording
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaStreamRef.current = stream;
        setIsRecording(true);
        setTranscription([]); // Clear previous transcription
        await setupAudioProcessing(stream);
      } catch (error) {
        console.error('Error accessing microphone:', error);
        setTranscription([{ speaker: "Error", text: "Could not access microphone." }]);
      }
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Real-time Speech-to-Text with Speaker Diarization</h1>
      </header>
      <main>
        <div className="controls">
          <button onClick={handleToggleRecording}>
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>
        </div>

        <div className="visualizer-container">
          <h2>Audio Visualizer</h2>
          <canvas ref={canvasRef} width="600" height="100" className="audio-visualizer"></canvas>
        </div>

        <div className="transcription-container">
          <h2>Transcription</h2>
          <div className="transcription-output">
            {transcription.length === 0 && !isRecording && <p>Click "Start Recording" to begin.</p>}
            {transcription.map((segment, index) => (
              <p key={index}>
                <strong>{segment.speaker}</strong>
                {segment.start_time !== undefined && segment.end_time !== undefined
                  ? ` (${segment.start_time.toFixed(2)}s - ${segment.end_time.toFixed(2)}s)`
                  : ''}: {segment.text}
              </p>
            ))}
          </div>
        </div>
      </main>
      <footer>
        <p style={{"fontSize": "0.8em", "color": "#777"}}>Note: Model loading on the backend may take time on first start. Ensure your Hugging Face token is configured if models require it.</p>
      </footer>
    </div>
  );
}

// Simple logger to avoid cluttering console if not needed
const logger = {
    info: (...args: any[]) => console.log('[INFO]', ...args),
    warn: (...args: any[]) => console.warn('[WARN]', ...args),
    error: (...args: any[]) => console.error('[ERROR]', ...args),
};

export default App;
