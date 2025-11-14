import sys
import threading
import asyncio
import websockets
import json
import traceback
import time
from queue import Queue, Empty
from difflib import SequenceMatcher
from transcriber_core import MicrophoneTranscriber, DesktopTranscriber

# Global queues for communication between threads
output_queue = Queue()

class TranscriptionDeduplicator:
    """Filters out overlapping/duplicate transcriptions with continuation detection"""
    
    def __init__(self, similarity_threshold=0.65, time_window=2.5, overlap_words=3):
        self.recent_transcripts = {}  # source -> (text, timestamp)
        self.similarity_threshold = similarity_threshold
        self.time_window = time_window
        self.overlap_words = overlap_words  # Min words to check for overlap
        
    def _get_overlap_length(self, prev_text, curr_text):
        """Calculate how many words overlap at the end of prev and start of curr"""
        prev_words = prev_text.lower().split()
        curr_words = curr_text.lower().split()
        
        max_check = min(len(prev_words), len(curr_words), 10)  # Check up to 10 words
        overlap_length = 0
        
        for i in range(1, max_check + 1):
            if prev_words[-i:] == curr_words[:i]:
                overlap_length = i
        
        return overlap_length
    
    def _is_substring_or_similar(self, prev_text, curr_text):
        """Check if curr_text is contained in prev_text or vice versa"""
        prev_lower = prev_text.lower().strip()
        curr_lower = curr_text.lower().strip()
        
        # Check substring
        if curr_lower in prev_lower or prev_lower in curr_lower:
            return True
        
        # Check high similarity
        ratio = SequenceMatcher(None, prev_lower, curr_lower).ratio()
        return ratio > self.similarity_threshold
    
    def process(self, text, source):
        """
        Process a transcription and decide whether to skip, merge, or output.
        Returns: (should_output, final_text)
        """
        current_time = time.time()
        
        # Get most recent from this source
        if source in self.recent_transcripts:
            prev_text, prev_timestamp = self.recent_transcripts[source]
            
            # Check if too old
            if current_time - prev_timestamp > self.time_window:
                # Old enough to be separate
                self.recent_transcripts[source] = (text, current_time)
                return True, text
            
            # Check for duplicate/substring
            if self._is_substring_or_similar(prev_text, text):
                # Skip if current is contained in previous
                if text.lower().strip() in prev_text.lower():
                    return False, None
                # Update if current contains previous (it's longer)
                elif prev_text.lower().strip() in text.lower():
                    self.recent_transcripts[source] = (text, current_time)
                    return True, text
                else:
                    # Very similar, skip
                    return False, None
            
            # Check for continuation (overlapping words)
            overlap_length = self._get_overlap_length(prev_text, text)
            
            if overlap_length >= self.overlap_words:
                # This is a continuation, merge them
                curr_words = text.split()
                
                # Remove overlapping words from current
                unique_part = " ".join(curr_words[overlap_length:])
                
                if unique_part.strip():
                    # Merge: previous + new unique part
                    merged_text = prev_text + " " + unique_part
                    self.recent_transcripts[source] = (merged_text, current_time)
                    return True, merged_text
                else:
                    # No new content, skip
                    return False, None
            
            # Not a continuation or duplicate, treat as new
            self.recent_transcripts[source] = (text, current_time)
            return True, text
        
        else:
            # First time seeing this source
            self.recent_transcripts[source] = (text, current_time)
            return True, text

# Initialize deduplicator for both microphone and desktop audio
deduplicator = TranscriptionDeduplicator(
    similarity_threshold=0.70,  # Higher threshold
    time_window=3.0,  # Longer window
    overlap_words=3   # Min 3 overlapping words to detect continuation
)

async def websocket_server(websocket):
    """Handles WebSocket communication with the main bot."""
    print("WebSocket client connected.")
    try:
        while True:
            try:
                message = output_queue.get(block=False)
                await websocket.send(json.dumps(message))
                output_queue.task_done()
            except Empty:
                await asyncio.sleep(0.01)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"An error occurred in the WebSocket server: {e}")
        traceback.print_exc()

def run_transcription_system(mic_transcriber, desktop_transcriber):
    """Starts the transcription threads and routes output to the queue."""
    print("🎙️ Initializing Dual Transcriber System (Fast Batch Mode)...")

    mic_thread = threading.Thread(
        target=mic_transcriber.run,
        daemon=True,
        name="MicThread"
    )
    desktop_thread = threading.Thread(
        target=desktop_transcriber.run,
        daemon=True,
        name="DesktopThread"
    )

    desktop_thread.start()
    time.sleep(1)
    mic_thread.start()

    print("✅ Both transcription systems are running.")
    print("   🎤 Microphone: FAST BATCH MODE (0.5s silence, deduplication enabled)")
    print("   🖥️ Desktop: CHUNKED MODE (deduplication enabled)")
    print("Waiting for transcriptions...")

    while True:
        try:
            mic_result = None
            desktop_result = None
            
            # Get from mic queue - WITH DEDUPLICATION (batch mode)
            try:
                text, filename, source, confidence = mic_transcriber.result_queue.get(block=False)
                
                # Process with deduplicator for microphone
                should_output, final_text = deduplicator.process(text, "microphone")
                
                if should_output:
                    mic_result = {
                        "source": "microphone",
                        "text": final_text,
                        "confidence": confidence,
                        "audio_type": "speech"
                    }
                mic_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Error processing mic result: {e}")
            
            # Get from desktop queue - WITH DEDUPLICATION (chunked mode)
            try:
                text, filename, audio_type, confidence = desktop_transcriber.result_queue.get(block=False)
                
                # Process with deduplicator (separate tracking for desktop)
                should_output, final_text = deduplicator.process(text, "desktop")
                
                if should_output:
                    desktop_result = {
                        "source": "desktop",
                        "text": final_text,
                        "confidence": confidence,
                        "audio_type": audio_type
                    }
                desktop_transcriber.result_queue.task_done()
            except Empty:
                pass
            except Exception as e:
                print(f"Error processing desktop result: {e}")
            
            # Output results
            if mic_result:
                print(f"[MIC] {mic_result['text']}", flush=True)
                output_queue.put(mic_result)
            if desktop_result:
                audio_type = desktop_result['audio_type'].upper()
                print(f"[DESKTOP-{audio_type}] {desktop_result['text']}", flush=True)
                output_queue.put(desktop_result)
                
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in transcription routing loop: {e}")
            traceback.print_exc()
            time.sleep(0.1)

async def main():
    """Main function to start the WebSocket and transcription systems."""
    print("="*60)
    print("Starting Nami Hearing App (Fast Batch Mode)")
    print("="*60)
    print("This app transcribes:")
    print("  • Microphone input: FAST BATCH (0.5s silence threshold)")
    print("  • Desktop audio: CHUNKED (classified as 'speech' or 'music')")
    print("\nMicrophone features:")
    print("  • Fast response: 0.5-0.8s latency")
    print("  • Accurate complete sentences")
    print("  • Smart deduplication")
    print("\nDesktop deduplication:")
    print("  • Filters duplicate transcriptions")
    print("  • Merges overlapping continuations")
    print("="*60)

    mic_transcriber = MicrophoneTranscriber()
    desktop_transcriber = DesktopTranscriber()

    transcription_thread = threading.Thread(
        target=run_transcription_system,
        args=(mic_transcriber, desktop_transcriber),
        daemon=True
    )
    transcription_thread.start()
    
    async with websockets.serve(websocket_server, "localhost", 8003):
        print("\n✅ WebSocket server listening on ws://localhost:8003")
        print("Ready to accept connections from main app...\n")
        
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
            mic_transcriber.stop_event.set()
            desktop_transcriber.stop_event.set()
            time.sleep(1)
            print("Goodbye!")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            mic_transcriber.stop_event.set()
            desktop_transcriber.stop_event.set()

if __name__ == "__main__":
    asyncio.run(main())