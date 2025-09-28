import os
import time
import numpy as np
import soundfile as sf
from threading import Thread

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
        """Processes a single audio chunk in a separate thread."""
        filename = None
        try:
            # Pre-process audio to filter out silence
            amplitude = np.abs(chunk).mean()
            if amplitude < 0.005:
                return

            filename = self.save_audio(chunk)
            
            if self.transcriber.auto_detect:
                audio_type, confidence = self.transcriber.classifier.classify(chunk)
            else:
                audio_type = self.transcriber.classifier.current_type
                confidence = 0.8
                
            if confidence < 0.4:
                return
                
            segments, info = self.transcriber.model.transcribe(chunk, beam_size=1, language="en")
            text = "".join(seg.text for seg in segments).strip()
            
            if text:
                self.transcriber.result_queue.put((text, filename, audio_type, confidence))
                
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