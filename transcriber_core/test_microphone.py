#!/usr/bin/env python3
"""
Test script to verify microphone transcription is working
"""

import numpy as np
import sounddevice as sd
import time
import sys
import os
import mlx.core as mx

try:
    # Attempt to import the new Parakeet library
    import parakeet_mlx
    PARAKEET_AVAILABLE = True
except ImportError:
    print("❌ 'parakeet-mlx' not found. Please ensure it is installed.")
    print("Run: pip install parakeet-mlx")
    PARAKEET_AVAILABLE = False
    
def test_microphone():
    """Test microphone input and transcription"""
    
    if not PARAKEET_AVAILABLE:
        print("Cannot proceed without 'parakeet-mlx'.")
        return False
        
    # Configuration
    SAMPLE_RATE = 16000
    DURATION = 3  # seconds
    # Assuming device ID from config, or fallback
    try:
        from nami.config import MICROPHONE_DEVICE_ID
        DEVICE_ID = MICROPHONE_DEVICE_ID
    except (ImportError, ModuleNotFoundError):
        DEVICE_ID = 4  # Your Scarlett Solo 4th Gen
        print(f"⚠️ Using fallback microphone device ID: {DEVICE_ID}")
    
    print("🎤 Microphone Test (using Parakeet)")
    print("=" * 30)
    
    # List available devices
    print("📱 Available input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            marker = " ← SELECTED" if i == DEVICE_ID else ""
            print(f"  {i}: {device['name']}{marker}")
    
    # Load the Parakeet model
    print(f"\n🤖 Loading Parakeet model...")
    try:
        model = parakeet_mlx.from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # Test recording
    print(f"\n🎤 Testing microphone recording...")
    print(f"   Device: {DEVICE_ID}")
    print(f"   Sample Rate: {SAMPLE_RATE} Hz")
    print(f"   Duration: {DURATION} seconds")
    
    try:
        print("\n🔴 Recording... Speak now!")
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            device=DEVICE_ID,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("⏹️ Recording finished")
        
        # Check audio level
        audio_flat = audio.flatten()
        rms_level = np.sqrt(np.mean(audio_flat**2))
        max_level = np.max(np.abs(audio_flat))
        
        print(f"\n📊 Audio Analysis:")
        print(f"   RMS Level: {rms_level:.4f}")
        print(f"   Max Level: {max_level:.4f}")
        print(f"   Length: {len(audio_flat)} samples ({len(audio_flat)/SAMPLE_RATE:.1f}s)")
        
        if rms_level < 0.001:
            print("⚠️ Audio level very low - check microphone connection/gain")
            return False
        elif rms_level < 0.01:
            print("⚠️ Audio level low - you might need to speak louder or increase gain")
        else:
            print("✅ Audio level good")
        
        # Test transcription with Parakeet
        print(f"\n🔄 Transcribing with Parakeet...")
        try:
            # CORRECTED: Call transcribe_stream from the model object without chunk_duration
            with model.transcribe_stream() as transcriber:
                transcriber.add_audio(mx.array(audio_flat))
                # Finalize the stream and get the result
                result = transcriber.result
            
            # Check if result is a valid object before trying to access its attributes
            if result and hasattr(result, 'text'):
                text = result.text.strip()
            else:
                text = ""

            print(f"\n📝 Transcription Results:")
            print(f"   Text: '{text}'")
            
            if text:
                print("✅ Transcription successful!")
                return True
            else:
                print("⚠️ No text transcribed - try speaking louder or closer to microphone")
                return False
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return False

def main():
    success = test_microphone()
    
    if success:
        print("\n🎉 Microphone test successful!")
        print("Your microphone should work with the full system.")
    else:
        print("\n❌ Microphone test failed.")
        print("Check your microphone connection and try adjusting the device ID in config.py")

if __name__ == "__main__":
    main()