# Nami Hearing App

This is a standalone application responsible for all audio transcription and classification for the PeepingNami bot. It listens to both the user's microphone and the desktop audio, transcribes the output in real-time, and sends the data to the main Nami bot application via a WebSocket server.

### Setup

1.  **Create a new Conda environment (optional but recommended):**
    ```bash
    conda create -n nami-hearing python=3.12
    conda activate nami-hearing
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python main.py
    ```

This app is designed to be run in parallel with the main `nami` bot application.