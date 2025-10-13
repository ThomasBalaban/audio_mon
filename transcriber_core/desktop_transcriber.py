import os
import time
import re
from threading import Thread, Event
from queue import Queue
import sounddevice as sd
import numpy as np
from .desktop_speech_music_classifier import SpeechMusicClassifier
from .desktop_audio_processor import AudioProcessor
from faster_whisper import WhisperModel
from .config import FS, MODEL_SIZE, DEVICE, SAVE_DIR, MAX_THREADS, COMPUTE_TYPE, DESKTOP_DEVICE_ID

# Define stop_event at the module level
stop_event = Event()

class SpeechMusicTranscriber:
    def __init__(self, keep_files=False, auto_detect=True, transcript_manager=None):
        self.FS = FS
        self.SAVE_DIR = SAVE_DIR
        self.MAX_THREADS = MAX_THREADS
        self.MODEL_SIZE = MODEL_SIZE
        self.DEVICE = DEVICE
        self.COMPUTE_TYPE = COMPUTE_TYPE
        self.DESKTOP_DEVICE_ID = DESKTOP_DEVICE_ID

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print(f"🎙️ Initializing faster-whisper for Desktop Audio: {self.MODEL_SIZE} on {self.DEVICE}")
        if self.DESKTOP_DEVICE_ID is not None:
            print(f"🔊 Using desktop audio device ID: {self.DESKTOP_DEVICE_ID}")
            try:
                device_info = sd.query_devices(self.DESKTOP_DEVICE_ID)
                print(f"   Device: {device_info['name']}")
                print(f"   Native Sample Rate: {device_info['default_samplerate']} Hz")
                print(f"   Will resample to: {self.FS} Hz")
            except Exception as e:
                print(f"   Warning: Could not get device info: {e}")
        else:
            print("🔊 Using default desktop audio device")
            
        try:
            self.model = WhisperModel(self.MODEL_SIZE, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
            print("✅ faster-whisper model for Desktop is ready.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        self.result_queue = Queue()
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

        # Name correction dictionary
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

    def run(self):
        """Starts the audio stream (NO output_worker thread!)."""
        print(f"Model: {self.MODEL_SIZE.upper()} | Device: {self.DEVICE.upper()}")
        print(f"Target Sample Rate: {self.FS} Hz")

        # REMOVED: output_thread - name correction now happens in process_chunk
        # All output will be handled by main.py

        try:
            device_info = sd.query_devices(self.DESKTOP_DEVICE_ID)
            native_samplerate = int(device_info['default_samplerate'])
            channels = min(device_info['max_input_channels'], 2)
            
            print(f"🎧 Opening audio stream:")
            print(f"   Device ID: {self.DESKTOP_DEVICE_ID}")
            print(f"   Native rate: {native_samplerate} Hz")
            print(f"   Channels: {channels}")
            print(f"   Resampling: {'YES' if native_samplerate != self.FS else 'NO'}")
            
            stream_kwargs = {
                'device': self.DESKTOP_DEVICE_ID,
                'samplerate': native_samplerate,
                'channels': channels,
                'callback': self.audio_processor.audio_callback,
                'blocksize': native_samplerate // 10,
                'dtype': 'float32'
            }
            
            with sd.InputStream(**stream_kwargs):
                print(f"🎧 Listening to desktop audio (device {self.DESKTOP_DEVICE_ID})...")
                print("   Waiting for audio...")
                
                last_activity_report = time.time()
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                    
                    if time.time() - last_activity_report > 30:
                        print(f"   [Status] Active threads: {self.active_threads}, Buffer size: {len(self.audio_processor.audio_buffer)}")
                        last_activity_report = time.time()

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping desktop transcriber...")
        except Exception as e:
            print(f"\n❌ Error starting desktop audio stream: {e}")
            import traceback
            traceback.print_exc()
            print("\nTroubleshooting tips:")
            print("1. Check if the device is properly connected")
            print("2. Try running: python helpers/sound_devices.py")
            print("3. Verify DESKTOP_DEVICE_ID in config.py")
        finally:
            self.stop_event.set()
            print("\nShutting down desktop transcriber...")
            if not self.keep_files:
                time.sleep(0.5)
                for filename in self.saved_files:
                     if os.path.exists(filename):
                        try: 
                            os.remove(filename)
                        except: 
                            pass
            print("🖥️ Desktop transcription stopped.")

def run_desktop_transcriber():
    """Main entry point function for hearing.py to call."""
    try:
        transcriber = SpeechMusicTranscriber()
        transcriber.run()
    except Exception as e:
        print(f"A critical error occurred in the desktop transcriber: {e}")