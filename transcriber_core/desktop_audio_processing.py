# nami/audio_utils/desktop_audio_processor.py
import os
import time
import numpy as np
import soundfile as sf
import re
from threading import Thread
from scipy import signal
import torch

class AudioProcessor:
    def __init__(self, transcriber):
        self.transcriber = transcriber
        self.audio_buffer = np.array([], dtype=np.float32)

    def audio_callback(self, indata, frames, timestamp, status):
        """Buffers audio and spawns processing threads for overlapping chunks."""
        try:
            from nami.config import FS, CHUNK_DURATION, OVERLAP, MAX_THREADS

            # Resume processing if thread count is low
            if not self.transcriber.processing_lock.is_set() and self.transcriber.active_threads < MAX_THREADS * 0.5:
                self.transcriber.processing_lock.set()
                    
            if self.transcriber.stop_event.is_set():
                return
                
            new_audio = indata.flatten().astype(np.float32)
            self.audio_buffer = np.concatenate([self.audio_buffer, new_audio])
            
            # Process when buffer reaches target duration and we're not overloaded
            if (self.transcriber.processing_lock.is_set() and 
                len(self.audio_buffer) >= FS * CHUNK_DURATION and
                self.transcriber.active_threads < MAX_THREADS):
                    
                # Copy chunk to prevent modification during processing
                chunk = self.audio_buffer[:FS*CHUNK_DURATION].copy()
                # Slide buffer forward to create overlap
                self.audio_buffer = self.audio_buffer[int(FS*(CHUNK_DURATION-OVERLAP)):]
                    
                self.transcriber.active_threads += 1
                Thread(target=self.process_chunk, args=(chunk,)).start()
                self.transcriber.last_processed = time.time()
                
                # If we get too many threads, temporarily pause processing
                if self.transcriber.active_threads >= MAX_THREADS:
                    self.transcriber.processing_lock.clear()
                    print(f"Pausing processing - too many active threads: {self.transcriber.active_threads}")
                    
        except Exception as e:
            print(f"Audio callback error: {e}")
            self.audio_buffer = np.array([], dtype=np.float32)
    
    def process_chunk(self, chunk):
        """Process audio chunk and transcribe based on type with enhanced error prevention"""
        filename = None
        try:
            # Exit early if we're stopping
            if self.transcriber.stop_event.is_set():
                return
                
            # Pre-process audio
            # 1. Ensure minimum length
            if len(chunk) < self.transcriber.FS * 0.5:
                return
                
            # 2. Apply noise gate - filter out very quiet audio completely
            amplitude = np.abs(chunk).mean()
            if amplitude < 0.005:  # Increased threshold for better noise rejection
                return
                
            # 3. Apply dynamic range compression to reduce background noise
            # Simple compression: reduce volume of quiet parts
            threshold = 0.02
            ratio = 0.5  # Compression ratio
            compressed = np.zeros_like(chunk)
            for i in range(len(chunk)):
                if abs(chunk[i]) > threshold:
                    compressed[i] = chunk[i]
                else:
                    compressed[i] = chunk[i] * ratio
            
            # 4. Normalize audio after compression
            max_val = np.max(np.abs(compressed))
            if max_val < 1e-10:  # Avoid division by zero
                return
                
            chunk = compressed / max_val
            
            # 5. Ensure even length
            if len(chunk) % 2 != 0:
                chunk = np.pad(chunk, (0, 1), 'constant')
            
            # Save audio file
            filename = self.save_audio(chunk)
            
            # Classify audio type if auto_detect is enabled
            if self.transcriber.auto_detect:
                audio_type, confidence = self.transcriber.classifier.classify(chunk)
            else:
                audio_type = self.transcriber.classifier.current_type
                confidence = 0.8
                
            # Skip processing completely if confidence is too low
            if confidence < 0.4:
                # Clean up file
                if not self.transcriber.keep_files and filename and os.path.exists(filename):
                    os.remove(filename)
                return
                
            # Transcribe with error handling - FIXED: Removed problematic parameters
            try:
                # Process and transcribe using the segments, info pattern
                segments, info = self.transcriber.model.transcribe(
                    chunk, 
                    beam_size=1, 
                    language="en"
                )
                text = "".join(seg.text for seg in segments).strip()
                
                # Post-process text - remove repeated characters (like B-B-B-B)
                text = re.sub(r'(\w)(\s*-\s*\1){3,}', r'\1...', text)
                
                # Handle empty or very short results based on audio type
                min_length = 2 if audio_type == "speech" else 4  # Higher threshold for music
                
                if text and len(text) >= min_length:
                    self.transcriber.result_queue.put((text, filename, audio_type, confidence))
                else:
                    # Silently clean up files with no text
                    if not self.transcriber.keep_files and filename and os.path.exists(filename):
                        os.remove(filename)
                    
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                
                # Clean up file on error
                if not self.transcriber.keep_files and filename and os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except:
                        pass
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
        finally:
            # Ensure the thread count is always decremented
            self.transcriber.active_threads -= 1
            # If the file was created but not used, clean it up
            if filename and not self.transcriber.keep_files and os.path.exists(filename):
                try:
                    # Check if it's already been queued for deletion
                    if not any(filename in item for item in list(self.transcriber.result_queue.queue)):
                         os.remove(filename)
                except:
                    pass

    def save_audio(self, chunk):
        """Saves audio chunk to file and returns filename."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.transcriber.SAVE_DIR, f"desktop_{timestamp}.wav")
        sf.write(filename, chunk, self.transcriber.FS, subtype='PCM_16')
        self.transcriber.saved_files.append(filename)
        return filename