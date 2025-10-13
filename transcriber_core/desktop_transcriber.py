import os
import time
import re
from threading import Thread, Event
from queue import Queue
import sounddevice as sd # type: ignore
from .desktop_speech_music_classifier import SpeechMusicClassifier
from .desktop_audio_processor import AudioProcessor
from faster_whisper import WhisperModel # type: ignore
from .config import FS, MODEL_SIZE, DEVICE, SAVE_DIR, MAX_THREADS, COMPUTE_TYPE, DESKTOP_DEVICE_ID # Added DESKTOP_DEVICE_ID

# --- FIX: Define stop_event at the module level so it can be imported ---
stop_event = Event()

class SpeechMusicTranscriber:
    def __init__(self, keep_files=False, auto_detect=True, transcript_manager=None):
        try:
            from nami.config import FS, MODEL_SIZE, DEVICE, SAVE_DIR, MAX_THREADS, DESKTOP_DEVICE_ID
            self.FS = FS
            self.SAVE_DIR = SAVE_DIR
            self.MAX_THREADS = MAX_THREADS
            self.MODEL_SIZE = "base.en"
            self.DEVICE = "cpu"
            self.COMPUTE_TYPE = "int8"
            self.DESKTOP_DEVICE_ID = DESKTOP_DEVICE_ID  # Store the desktop device ID
        except ImportError:
            self.FS = 16000
            self.SAVE_DIR = "audio_captures"
            self.MAX_THREADS = 4
            self.MODEL_SIZE = "base.en"
            self.DEVICE = "cpu"
            self.COMPUTE_TYPE = "int8"
            self.DESKTOP_DEVICE_ID = None  # Default to None if config not available
            print("⚠️ Using fallback config values for transcriber")

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print(f"🎙️ Initializing faster-whisper for Desktop Audio: {self.MODEL_SIZE} on {self.DEVICE}")
        if self.DESKTOP_DEVICE_ID is not None:
            print(f"🔊 Using desktop audio device ID: {self.DESKTOP_DEVICE_ID}")
        else:
            print("🔊 Using default desktop audio device")
            
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
            # FIXED: Add device parameter to use the specified desktop audio device
            stream_kwargs = {
                'samplerate': FS,
                'channels': 1,
                'callback': self.audio_processor.audio_callback,
                'blocksize': FS//10
            }
            
            # Only add device parameter if we have a specific device ID
            if self.DESKTOP_DEVICE_ID is not None:
                stream_kwargs['device'] = self.DESKTOP_DEVICE_ID
            
            with sd.InputStream(**stream_kwargs):
                if self.DESKTOP_DEVICE_ID is not None:
                    print(f"🎧 Listening to desktop audio on device {self.DESKTOP_DEVICE_ID}...")
                else:
                    print("🎧 Listening to desktop audio on default device...")
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        except Exception as e:
            print(f"\n❌ Error starting desktop audio stream: {e}")
            print("This might be due to an invalid device ID or device not available.")
            # Try to list available devices to help debug
            try:
                print("\nAvailable audio devices:")
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        marker = " ← CONFIGURED" if i == self.DESKTOP_DEVICE_ID else ""
                        print(f"  {i}: {device['name']}{marker}")
            except:
                print("Could not list audio devices.")
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