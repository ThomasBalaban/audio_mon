# nami/audio_utils/microphone.py
import os
import sys
import numpy as np
import sounddevice as sd
import time
import soundfile as sf
import re
import traceback
from queue import Queue
from threading import Thread, Event, Lock
# We will not import transcribe_stream directly, as it's not at the top level
import parakeet_mlx
import mlx.core as mx
from ..config import MICROPHONE_DEVICE_ID, FS, CHUNK_DURATION, OVERLAP, MAX_THREADS, SAVE_DIR

# Configuration for Microphone
SAMPLE_RATE = FS  # Use the same sample rate as desktop audio
CHANNELS = 1

# --- VAD (Voice Activity Detection) Parameters ---
VAD_ENERGY_THRESHOLD = 0.008  # Energy threshold to start detecting speech
VAD_SILENCE_DURATION = 1.5    # Seconds of silence to consider speech ended
VAD_MAX_SPEECH_DURATION = 15.0 # Maximum seconds to buffer before forcing a transcription

# Global variables for this module
stop_event = Event()

class MicrophoneTranscriber:
    """Improved microphone transcriber with VAD, smart buffering, and name correction"""

    def __init__(self, keep_files=False, transcript_manager=None):
        self.FS = SAMPLE_RATE
        self.SAVE_DIR = SAVE_DIR
        self.MAX_THREADS = MAX_THREADS
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print("🎙️ Initializing parakeet-mlx for Microphone...")
        try:
            self.model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
            print("✅ parakeet-mlx for Microphone is ready.")
        except Exception as e:
            print(f"❌ Error initializing Parakeet: {e}")
            raise

        self.result_queue = Queue()
        self.stop_event = stop_event
        self.saved_files = []
        self.keep_files = keep_files
        self.active_threads = 0
        self.processing_lock = Event()
        self.processing_lock.set()

        # --- VAD and Buffering State ---
        self.speech_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_start_time = None
        self.speech_start_time = None
        self.buffer_lock = Lock()

        self.transcript_manager = transcript_manager

        # --- MODIFIED: Expanded Name Correction Dictionary ---
        self.name_variations = {
            r'\bnaomi\b': 'Nami',
            r'\bnow may\b': 'Nami',
            r'\bnomi\b': 'Nami',
            r'\bnamy\b': 'Nami',
            r'\bnot me\b': 'Nami',
            r'\bnah me\b': 'Nami',
            r'\bnonny\b': 'Nami',
            r'\bnonni\b': 'Nami',
            r'\bmamie\b': 'Nami',
            r'\bgnomey\b': 'Nami',
            r'\barmy\b': 'Nami',
            r'\bpeepingnaomi\b': 'PeepingNami',
            r'\bpeepingnomi\b': 'PeepingNami'
        }


    def audio_callback(self, indata, frames, timestamp, status):
        """Analyzes audio for speech, buffers it, and sends complete utterances for transcription."""
        # --- ADDED: Enhanced Error Reporting ---
        if status:
            print(f"[MIC-ERROR] Audio callback status: {status}", file=sys.stderr)
            # If we get an input overflow, we should probably clear the buffer to recover.
            if status.input_overflow:
                print("[MIC-WARN] Input overflow detected, clearing buffer.", file=sys.stderr)
                with self.buffer_lock:
                    self.speech_buffer = np.array([], dtype=np.float32)

        if self.stop_event.is_set():
            return

        new_audio = indata.flatten().astype(np.float32)
        rms_amplitude = np.sqrt(np.mean(new_audio**2))

        with self.buffer_lock:
            if rms_amplitude > VAD_ENERGY_THRESHOLD:
                # --- Speech Detected ---
                if not self.is_speaking:
                    print("🎤 Speech detected, starting to buffer...")
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                self.speech_buffer = np.concatenate([self.speech_buffer, new_audio])
                self.silence_start_time = None # Reset silence timer

                # --- Smart Overflow Protection ---
                if time.time() - self.speech_start_time > VAD_MAX_SPEECH_DURATION:
                    print("🎤 Long speech detected, processing current buffer...")
                    self._process_speech_buffer()

            elif self.is_speaking:
                # --- Silence after speech ---
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()

                if time.time() - self.silence_start_time > VAD_SILENCE_DURATION:
                    print("🎤 Silence detected, processing buffered speech...")
                    self._process_speech_buffer()

    def _process_speech_buffer(self):
        """Processes the buffered speech in a separate thread."""
        if len(self.speech_buffer) > self.FS * 0.5 and self.active_threads < self.MAX_THREADS: # At least 0.5s of audio
            chunk_to_process = self.speech_buffer.copy()
            self.speech_buffer = np.array([], dtype=np.float32) # Clear buffer
            self.is_speaking = False
            self.silence_start_time = None
            self.speech_start_time = None

            self.active_threads += 1
            Thread(target=self.process_chunk, args=(chunk_to_process,)).start()
        else:
            # Discard very short utterances (noise)
            self.speech_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False

    def process_chunk(self, chunk):
        """Transcribes a chunk of audio."""
        filename = None
        try:
            filename = self.save_audio(chunk)
            
            # --- MODIFIED: Use the correct method call and arguments ---
            with self.model.transcribe_stream() as transcriber:
                transcriber.add_audio(mx.array(chunk))
                result = transcriber.result
                text = result.text.strip() if result and hasattr(result, 'text') else ""
            
            if text and len(text) >= 2:
                # We don't have a confidence from parakeet-mlx, so we'll set a default
                confidence = 0.9
                self.result_queue.put((text, filename, "microphone", confidence))
            else:
                if not self.keep_files and filename and os.path.exists(filename):
                    os.remove(filename)
        except Exception as e:
            # --- ADDED: More Detailed Error Logging ---
            print(f"[MIC-ERROR] Transcription thread failed: {str(e)}", file=sys.stderr)
            traceback.print_exc() # Print the full traceback for debugging
            if not self.keep_files and filename and os.path.exists(filename):
                try:
                    os.remove(filename)
                except:
                    pass
        finally:
            self.active_threads -= 1


    def save_audio(self, chunk):
        """Save audio chunk to file and return filename"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.SAVE_DIR, f"microphone_{timestamp}.wav")
        sf.write(filename, chunk, self.FS, subtype='PCM_16')
        self.saved_files.append(filename)
        return filename

    def output_worker(self):
        """Process and display transcription results"""
        while not self.stop_event.is_set():
            try:
                if not self.result_queue.empty():
                    text, filename, audio_type, confidence = self.result_queue.get()

                    if text:
                        corrected_text = text
                        for variation, name in self.name_variations.items():
                            corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)

                        # Print in the required format
                        print(f"[Microphone Input] {corrected_text}", flush=True)

                        # Clean up file after processing
                        if not self.keep_files and filename and os.path.exists(filename):
                            try:
                                os.remove(filename)
                            except Exception as e:
                                print(f"Error removing file: {str(e)}")

                    self.result_queue.task_done()
                time.sleep(0.05)
            except Exception as e:
                print(f"Microphone output worker error: {str(e)}")

    def run(self):
        """Start the audio stream and worker threads"""
        output_thread = Thread(target=self.output_worker, daemon=True)
        output_thread.start()

        try:
            blocksize = self.FS // 20  # 50ms blocks for responsive VAD

            with sd.InputStream(
                device=MICROPHONE_DEVICE_ID,
                samplerate=self.FS,
                channels=CHANNELS,
                callback=self.audio_callback,
                blocksize=blocksize,
                dtype='float32'
            ):
                print("🎤 Listening to microphone with VAD...")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping microphone transcriber...")
        # --- ADDED: Catching specific sounddevice errors ---
        except sd.PortAudioError as e:
            print(f"\n[MIC-FATAL] A PortAudio error occurred: {e}", file=sys.stderr)
            print("This could be due to a disconnected device or a driver issue.", file=sys.stderr)
        except Exception as e:
            print(f"\n[MIC-FATAL] An unexpected error occurred in the run loop: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.stop_event.set()
            print("\nShutting down microphone transcriber...")
            if not self.keep_files:
                time.sleep(0.5)
                for filename in self.saved_files:
                     if os.path.exists(filename):
                        try:
                            os.remove(filename)
                        except:
                            pass
            print("🎤 Microphone transcription stopped.")


def transcribe_microphone():
    """Main entry point function for hearing.py to call"""
    try:
        transcriber = MicrophoneTranscriber()
        transcriber.run()
    except Exception as e:
        print(f"A critical error occurred in the microphone transcriber: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        transcribe_microphone()
    except KeyboardInterrupt:
        print("\nStopping microphone listener...")
        stop_event.set()