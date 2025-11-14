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
import parakeet_mlx
import mlx.core as mx
from .config import (
    MICROPHONE_DEVICE_ID, 
    FS, 
    SAVE_DIR,
    STREAM_UPDATE_INTERVAL,
    SESSION_MAX_DURATION,
    SESSION_RESET_SILENCE,
    VAD_THRESHOLD
)

# Configuration for Microphone
SAMPLE_RATE = FS
CHANNELS = 1

# Global variables for this module
stop_event = Event()

class MicrophoneTranscriber:
    """Streaming microphone transcriber that sends updates continuously as speech is detected"""

    def __init__(self, keep_files=False, transcript_manager=None):
        self.FS = SAMPLE_RATE
        self.SAVE_DIR = SAVE_DIR
        
        os.makedirs(self.SAVE_DIR, exist_ok=True)

        print("🎙️ Initializing parakeet-mlx for Microphone (Streaming Mode)...")
        try:
            self.model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
            print("✅ parakeet-mlx for Microphone is ready.")
        except Exception as e:
            print(f"❌ Error initializing Parakeet: {e}")
            raise

        self.result_queue = Queue()
        self.stop_event = stop_event
        self.saved_files = []
        self.keep_files = keep_files

        # Streaming session state
        self.transcriber = None
        self.session_lock = Lock()
        self.session_start_time = None
        self.last_speech_time = None
        self.last_sent_text = ""
        
        # Audio buffering for session management
        self.audio_buffer = []
        self.chunk_buffer = np.array([], dtype=np.float32)  # Buffer for feeding larger chunks
        self.last_feed_time = time.time()
        self.is_speaking = False

        self.transcript_manager = transcript_manager

        # Name Correction Dictionary
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

    def apply_name_correction(self, text):
        """Apply name corrections to transcribed text"""
        corrected = text
        for variation, name in self.name_variations.items():
            corrected = re.sub(variation, name, corrected, flags=re.IGNORECASE)
        return corrected

    def should_reset_session(self):
        """Check if session should be reset based on duration and silence"""
        if self.session_start_time is None:
            return False
        
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        
        # Check if we've exceeded max session duration
        if session_duration < SESSION_MAX_DURATION:
            return False
        
        # Check if we have enough silence
        if self.last_speech_time is None:
            return False
            
        silence_duration = current_time - self.last_speech_time
        return silence_duration >= SESSION_RESET_SILENCE

    def reset_session(self):
        """Reset the transcription session"""
        with self.session_lock:
            if self.transcriber is not None:
                try:
                    # Get final result before closing
                    if hasattr(self.transcriber, 'result'):
                        result = self.transcriber.result
                        if result and hasattr(result, 'text'):
                            final_text = result.text.strip()
                            if final_text and final_text != self.last_sent_text:
                                corrected_text = self.apply_name_correction(final_text)
                                self.result_queue.put((corrected_text, None, "microphone", 0.85))
                                self.last_sent_text = final_text
                except Exception as e:
                    print(f"[MIC] Error getting final result: {e}")
                
                # Close the session
                try:
                    self.transcriber.__exit__(None, None, None)
                except:
                    pass
                    
                self.transcriber = None
                
            self.session_start_time = None
            self.last_sent_text = ""
            self.audio_buffer = []
            print("[MIC] Session reset")

    def ensure_session(self):
        """Ensure a transcription session is active"""
        with self.session_lock:
            if self.transcriber is None:
                try:
                    # Add context_size parameter for proper streaming
                    self.transcriber = self.model.transcribe_stream(
                        context_size=(256, 256),
                        depth=2,  # More encoder layers for consistency (try 2, 3, or 4)
                        keep_original_attention=False  # Keep as False for streaming
                    ).__enter__()
                    self.session_start_time = time.time()
                    self.last_sent_text = ""
                    self.chunk_buffer = np.array([], dtype=np.float32)
                    self.last_feed_time = time.time()
                    print("[MIC] New streaming session started with context_size=(256, 256)")
                except Exception as e:
                    print(f"[MIC-ERROR] Failed to create session: {e}")
                    self.transcriber = None

    def audio_callback(self, indata, frames, timestamp, status):
        """Process incoming audio and feed to streaming transcriber"""
        if status:
            if status.input_overflow:
                print("[MIC-WARN] Input overflow detected", file=sys.stderr)

        if self.stop_event.is_set():
            return

        new_audio = indata.flatten().astype(np.float32)
        rms_amplitude = np.sqrt(np.mean(new_audio**2))

        # Voice Activity Detection
        if rms_amplitude > VAD_THRESHOLD:
            self.is_speaking = True
            self.last_speech_time = time.time()
            
            # Ensure we have an active session
            self.ensure_session()
            
            # Buffer audio into larger chunks (feed every 0.5 seconds)
            if self.transcriber is not None:
                try:
                    with self.session_lock:
                        self.chunk_buffer = np.concatenate([self.chunk_buffer, new_audio])
                        self.audio_buffer.append(new_audio)
                        
                        # Feed to transcriber when we have ~0.5 seconds of audio
                        current_time = time.time()
                        if len(self.chunk_buffer) >= self.FS * 1.0 or (current_time - self.last_feed_time) >= 1.0:
                            if len(self.chunk_buffer) > 0:
                                self.transcriber.add_audio(mx.array(self.chunk_buffer))
                                self.chunk_buffer = np.array([], dtype=np.float32)
                                self.last_feed_time = current_time
                except Exception as e:
                    print(f"[MIC-ERROR] Failed to add audio: {e}")
                    # Reset session on error
                    self.transcriber = None
        else:
            self.is_speaking = False
            
            # Feed any remaining buffered audio when speech stops
            if self.transcriber is not None and len(self.chunk_buffer) > 0:
                try:
                    with self.session_lock:
                        self.transcriber.add_audio(mx.array(self.chunk_buffer))
                        self.chunk_buffer = np.array([], dtype=np.float32)
                        self.last_feed_time = time.time()
                except Exception as e:
                    print(f"[MIC-ERROR] Failed to add final audio: {e}")

    def result_polling_thread(self):
        """Continuously poll for new transcription results and send them"""
        print("[MIC] Result polling thread started")
        
        while not self.stop_event.is_set():
            try:
                # Check if we should reset session
                if self.should_reset_session():
                    self.reset_session()
                
                # Get current result if we have an active session
                if self.transcriber is not None:
                    try:
                        with self.session_lock:
                            if hasattr(self.transcriber, 'result'):
                                result = self.transcriber.result
                                
                                if result and hasattr(result, 'text'):
                                    current_text = result.text.strip()
                                    
                                    # Only send if text has changed
                                    if current_text and current_text != self.last_sent_text:
                                        # Apply name correction
                                        corrected_text = self.apply_name_correction(current_text)
                                        
                                        # Send to queue: (text, filename, source, confidence)
                                        self.result_queue.put((corrected_text, None, "microphone", 0.85))
                                        
                                        self.last_sent_text = current_text
                    except Exception as e:
                        print(f"[MIC-ERROR] Error getting result: {e}")
                        
            except Exception as e:
                print(f"[MIC-ERROR] Polling thread error: {e}")
                traceback.print_exc()
            
            # Sleep for the configured interval
            time.sleep(STREAM_UPDATE_INTERVAL)
        
        print("[MIC] Result polling thread stopped")

    def run(self):
        """Start the audio stream and result polling thread"""
        try:
            device_info = sd.query_devices(MICROPHONE_DEVICE_ID)
            print(f"\n🎤 Microphone Configuration (Streaming Mode):")
            print(f"   Device ID: {MICROPHONE_DEVICE_ID}")
            print(f"   Device: {device_info['name']}")
            print(f"   Sample Rate: {self.FS} Hz")
            print(f"   VAD Threshold: {VAD_THRESHOLD}")
            print(f"   Update Interval: {STREAM_UPDATE_INTERVAL}s")
            print(f"   Session Max Duration: {SESSION_MAX_DURATION}s")
            print(f"   Session Reset Silence: {SESSION_RESET_SILENCE}s")
        except Exception as e:
            print(f"⚠️ Could not get device info: {e}")

        # Start result polling thread
        polling_thread = Thread(
            target=self.result_polling_thread,
            daemon=True,
            name="MicResultPolling"
        )
        polling_thread.start()

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
                print("🎤 Listening to microphone in streaming mode...")
                print("   Speak to see live transcription\n")
                
                while not self.stop_event.is_set():
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nReceived interrupt, stopping microphone transcriber...")
        except sd.PortAudioError as e:
            print(f"\n[MIC-FATAL] A PortAudio error occurred: {e}", file=sys.stderr)
            print("This could be due to a disconnected device or a driver issue.", file=sys.stderr)
        except Exception as e:
            print(f"\n[MIC-FATAL] An unexpected error occurred in the run loop: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.stop_event.set()
            
            # Clean up session
            self.reset_session()
            
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