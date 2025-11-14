# audio_mon/transcriber_core/config.py

# Audio Settings
FS = 16000  # Sample rate in Hz
CHUNK_DURATION = 5  # Duration of each audio chunk in seconds
OVERLAP = 1.5  # Overlap between chunks in seconds
MAX_THREADS = 4  # Maximum number of threads for transcription
SAVE_DIR = "audio_captures"  # Directory for saving audio files
DESKTOP_DEVICE_ID = 4

# Microphone Settings
MICROPHONE_DEVICE_ID = 5  # Your Scarlett Solo 4th Gen

# Whisper Model Settings
MODEL_SIZE = "base.en"  # The whisper model to use
DEVICE = "cpu"  # The device to run the model on
COMPUTE_TYPE = "int8"  # Compute type for the model
