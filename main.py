import sys
import threading
import asyncio
import websockets
import json
import traceback
import time
from queue import Queue, Empty
from transcriber_core import MicrophoneTranscriber, DesktopTranscriber

# Global queues for communication between threads
output_queue = Queue()

async def websocket_server(websocket):
    """Handles WebSocket communication with the main bot."""
    print("WebSocket client connected.")
    try:
        while True:
            try:
                # Wait for data from the transcription threads
                message = output_queue.get(block=False)
                
                # Send the message
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
    print("🎙️ Initializing Dual Transcriber System...")

    # Create threads
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

    # Start the threads
    desktop_thread.start()
    time.sleep(1)  # Give desktop a moment to initialize
    mic_thread.start()

    print("✅ Both transcription systems are running.")
    print("Waiting for transcriptions...")

    # This thread simply pushes results to the output queue
    while True:
        try:
            # Use non-blocking get with timeout to check both queues efficiently
            mic_result = None
            desktop_result = None
            
            # Try to get from mic queue (non-blocking)
            try:
                text, filename, source, confidence = mic_transcriber.result_queue.get(block=False)
                mic_result = {
                    "source": "microphone",
                    "text": text,
                    "confidence": confidence,
                    "audio_type": "speech"
                }
                mic_transcriber.result_queue.task_done()
            except:
                pass  # Queue empty, that's fine
            
            # Try to get from desktop queue (non-blocking)
            try:
                text, filename, audio_type, confidence = desktop_transcriber.result_queue.get(block=False)
                desktop_result = {
                    "source": "desktop",
                    "text": text,
                    "confidence": confidence,
                    "audio_type": audio_type
                }
                desktop_transcriber.result_queue.task_done()
            except:
                pass  # Queue empty, that's fine
            
            # Put results in output queue and print to console
            if mic_result:
                print(f"[MIC] {mic_result['text']}", flush=True)
                output_queue.put(mic_result)
            if desktop_result:
                audio_type = desktop_result['audio_type'].upper()
                print(f"[DESKTOP-{audio_type}] {desktop_result['text']}", flush=True)
                output_queue.put(desktop_result)
                
            # Short sleep to prevent CPU spinning
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in transcription routing loop: {e}")
            traceback.print_exc()
            time.sleep(0.1)

async def main():
    """Main function to start the WebSocket and transcription systems."""
    print("="*60)
    print("Starting Nami Hearing App")
    print("="*60)
    print("This app transcribes:")
    print("  • Microphone input (always 'speech')")
    print("  • Desktop audio (classified as 'speech' or 'music')")
    print("\nOutput format via WebSocket:")
    print('  {"source": "microphone|desktop", "text": "...", ')
    print('   "confidence": 0.0-1.0, "audio_type": "speech|music"}')
    print("="*60)

    # Initialize both transcribers
    mic_transcriber = MicrophoneTranscriber()
    desktop_transcriber = DesktopTranscriber()

    # Start the transcription system in a separate thread
    transcription_thread = threading.Thread(
        target=run_transcription_system,
        args=(mic_transcriber, desktop_transcriber),
        daemon=True
    )
    transcription_thread.start()
    
    # Start the WebSocket server on the main thread
    async with websockets.serve(websocket_server, "localhost", 8003):
        print("\n✅ WebSocket server listening on ws://localhost:8003")
        print("Ready to accept connections from main app...\n")
        
        try:
            # Keep the server running
            await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down gracefully...")
            # Signal the transcriber threads to stop
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