import sys
import threading
import asyncio
import websockets
import json
import traceback
import time
from queue import Queue, Empty
from transcriber_core import MicrophoneTranscriber, DesktopTranscriber
from transcriber_core.config import FS

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
                await websocket.send(json.dumps(message))
                output_queue.task_done()
            except Empty:
                await asyncio.sleep(0.01)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        print(f"An error occurred in the WebSocket server: {e}")

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
    mic_thread.start()
    desktop_thread.start()

    print("✅ Both transcription systems are running.")
    print("Waiting for transcriptions...")

    # This thread simply pushes results to the output queue
    while True:
        try:
            # Check mic queue
            if not mic_transcriber.result_queue.empty():
                text, filename, audio_type, confidence = mic_transcriber.result_queue.get(timeout=0.1)
                output_queue.put({
                    "source": "microphone",
                    "text": text,
                    "confidence": confidence,
                })
                mic_transcriber.result_queue.task_done()

            # Check desktop queue
            if not desktop_transcriber.result_queue.empty():
                text, filename, audio_type, confidence = desktop_transcriber.result_queue.get(timeout=0.1)
                output_queue.put({
                    "source": "desktop",
                    "text": text,
                    "confidence": confidence,
                    "audio_type": audio_type
                })
                desktop_transcriber.result_queue.task_done()
                
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in transcription routing loop: {e}")
            break

async def main():
    """Main function to start the WebSocket and transcription systems."""
    print("Starting Nami Hearing App...")

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
        print("WebSocket server listening on ws://localhost:8003")
        
        try:
            # Keep the server running
            await asyncio.Future()  # run forever
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            # Signal the transcriber threads to stop
            mic_transcriber.stop_event.set()
            desktop_transcriber.stop_event.set()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            mic_transcriber.stop_event.set()
            desktop_transcriber.stop_event.set()

if __name__ == "__main__":
    asyncio.run(main())