import os
import time
import re # Import the regular expression module
from threading import Thread, Event
from queue import Queue
import sounddevice as sd
from .desktop_speech_music_classifier import SpeechMusicClassifier
from .desktop_audio_processor import AudioProcessor
from faster_whisper import WhisperModel

# --- FIX: Define stop_event at the module level so it can be imported ---
stop_event = Event()

class SpeechMusicTranscriber:
    def __init__(self, keep_files=False, auto_detect=True, transcript_manager=None):
        try:
            from nami.config import FS, MODEL_SIZE, DEVICE, SAVE_DIR, MAX_THREADS
            self.FS = FS
            self.SAVE_DIR = SAVE_DIR
            self.MAX_THREADS = MAX_THREADS
            self.MODEL_SIZE = "base.en"
            self.DEVICE = "cpu"
            self.COMPUTE_TYPE = "int8"
        except ImportError:
            self.FS = 16000
            self.SAVE_DIR = "audio_captures"
            self.MAX_THREADS = 4
            self.MODEL_SIZE = "base.en"
            self.DEVICE = "cpu"
            self.COMPUTE_TYPE = "int8"
            print("⚠️ Using fallback config values for transcriber")

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print(f"🎙️ Initializing faster-whisper for Desktop Audio: {self.MODEL_SIZE} on {self.DEVICE}")
        try:
            self.model = WhisperModel(self.MODEL_SIZE, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
            print("✅ faster-whisper model for Desktop is ready.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        self.result_queue = Queue()
        # --- FIX: Use the global stop_event ---
        self.stop_event = stop_event
        self.saved_files = []
        self.keep_files = keep_files
        self.active_threads = 0
        self.processing_lock = Event()
        self.processing_lock.set()
        self.last_processed = time.time()

        self.classifier = SpeechMusicClassifier()
        self.auto_detect = auto_detect
        self.transcript_manager = transcript_manager
        self.audio_processor = AudioProcessor(self)

        # --- ADDED: Name Correction Dictionary ---
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

    def output_worker(self):
        """Processes and displays transcription results."""
        while not self.stop_event.is_set():
            try:
                if not self.result_queue.empty():
                    text, filename, audio_type, confidence = self.result_queue.get()

                    if text:
                        # --- ADDED: Apply name correction ---
                        corrected_text = text
                        for variation, name in self.name_variations.items():
                            corrected_text = re.sub(variation, name, corrected_text, flags=re.IGNORECASE)

                        # Print in the required format using the corrected text
                        print(f"[{audio_type.upper()} {confidence:.2f}] {corrected_text}", flush=True)

                        # Clean up file after processing
                        if not self.keep_files and filename and os.path.exists(filename):
                            try: os.remove(filename)
                            except Exception as e: print(f"Error removing file: {str(e)}")

                    self.result_queue.task_done()
                time.sleep(0.05)
            except Exception as e:
                print(f"Output worker error: {str(e)}")

    def run(self):
        """Starts the audio stream and worker threads."""
        from nami.config import CHUNK_DURATION, OVERLAP, FS

        print(f"Model: {self.MODEL_SIZE.upper()} | Device: {self.DEVICE.upper()}")
        print(f"Chunk: {CHUNK_DURATION}s with {OVERLAP}s overlap")

        output_thread = Thread(target=self.output_worker, daemon=True)
        output_thread.start()

        try:
            with sd.InputStream(
                samplerate=FS,
                channels=1,
                callback=self.audio_processor.audio_callback,
                blocksize=FS//10
            ):
                print("🎧 Listening to desktop audio...")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        finally:
            self.stop_event.set()
            print("\nShutting down desktop transcriber...")
            if not self.keep_files:
                time.sleep(0.5)
                for filename in self.saved_files:
                     if os.path.exists(filename):
                        try: os.remove(filename)
                        except: pass
            print("🖥️ Desktop transcription stopped.")

def run_desktop_transcriber():
    """Main entry point function for hearing.py to call."""
    try:
        transcriber = SpeechMusicTranscriber()
        transcriber.run()
    except Exception as e:
        print(f"A critical error occurred in the desktop transcriber: {e}")