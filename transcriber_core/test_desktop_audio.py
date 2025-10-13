#!/usr/bin/env python3
"""
Test script to verify desktop audio capture is working
"""

import numpy as np
import sounddevice as sd
import time
import sys
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def list_audio_devices():
    """List all available audio devices"""
    print("🔊 Available Audio Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
            input_devices.append(i)
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
            
        print(f"ID {i:2d}: {device['name']}")
        print(f"      Type: {' & '.join(device_type)}")
        print(f"      Max Input Channels: {device['max_input_channels']}")
        print(f"      Max Output Channels: {device['max_output_channels']}")
        print(f"      Default Sample Rate: {device['default_samplerate']} Hz")
        print()
    
    return input_devices

def test_device_recording(device_id, duration=5):
    """Test recording from a specific device"""
    print(f"\n🎵 Testing Device ID {device_id}")
    print("=" * 40)
    
    try:
        device_info = sd.query_devices(device_id)
        print(f"Device: {device_info['name']}")
        print(f"Max Input Channels: {device_info['max_input_channels']}")
        print(f"Default Sample Rate: {device_info['default_samplerate']} Hz")
        
        if device_info['max_input_channels'] == 0:
            print("❌ This device has no input channels!")
            return False
            
        # Use device's default sample rate
        sample_rate = int(device_info['default_samplerate'])
        
        print(f"\n🔴 Recording {duration} seconds from device {device_id}...")
        print("   Play some audio on your computer now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_id,
            dtype='float32'
        )
        
        # Show a countdown
        for i in range(duration, 0, -1):
            print(f"   Recording... {i} seconds remaining", end='\r')
            time.sleep(1)
        
        sd.wait()  # Wait until recording is finished
        print("\n⏹️  Recording finished!")
        
        # Analyze the audio
        audio_flat = audio_data.flatten()
        rms_level = np.sqrt(np.mean(audio_flat**2))
        max_level = np.max(np.abs(audio_flat))
        
        print(f"\n📊 Audio Analysis:")
        print(f"   Length: {len(audio_flat)} samples ({len(audio_flat)/sample_rate:.1f}s)")
        print(f"   RMS Level: {rms_level:.6f}")
        print(f"   Max Level: {max_level:.6f}")
        print(f"   Dynamic Range: {20*np.log10(max_level/rms_level + 1e-10):.1f} dB")
        
        # Classify the result
        if rms_level < 0.0001:
            print("❌ NO AUDIO DETECTED - This device is not picking up any sound")
            status = "silent"
        elif rms_level < 0.001:
            print("⚠️  Very quiet audio detected - might be background noise")
            status = "very_quiet"
        elif rms_level < 0.01:
            print("🟡 Quiet audio detected - could be distant or low volume")
            status = "quiet"
        elif rms_level < 0.1:
            print("✅ Good audio level detected!")
            status = "good"
        else:
            print("🔥 Very loud audio detected!")
            status = "loud"
        
        # Save the audio file for inspection
        filename = f"desktop_test_device_{device_id}_{int(time.time())}.wav"
        
        # Convert to 16-bit for saving
        audio_int16 = (audio_flat * 32767).astype(np.int16)
        write(filename, sample_rate, audio_int16)
        print(f"💾 Audio saved as: {filename}")
        
        # Create a simple visualization
        try:
            plt.figure(figsize=(12, 6))
            
            # Time domain plot
            plt.subplot(2, 1, 1)
            time_axis = np.linspace(0, duration, len(audio_flat))
            plt.plot(time_axis, audio_flat)
            plt.title(f'Desktop Audio Test - Device {device_id}: {device_info["name"]}')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.grid(True)
            
            # Frequency domain plot
            plt.subplot(2, 1, 2)
            fft_data = np.fft.rfft(audio_flat)
            fft_freqs = np.fft.rfftfreq(len(audio_flat), 1/sample_rate)
            plt.plot(fft_freqs, 20*np.log10(np.abs(fft_data) + 1e-10))
            plt.title('Frequency Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.grid(True)
            plt.xlim(0, 8000)  # Show up to 8kHz
            
            plt.tight_layout()
            plot_filename = f"desktop_test_device_{device_id}_plot.png"
            plt.savefig(plot_filename)
            print(f"📊 Visualization saved as: {plot_filename}")
            plt.close()
            
        except Exception as e:
            print(f"Could not create visualization: {e}")
        
        return status
        
    except Exception as e:
        print(f"❌ Error testing device {device_id}: {e}")
        return False

def test_desktop_audio_config():
    """Test the specific desktop audio device from config"""
    try:
        # Try to import the config
        try:
            from transcriber_core.config import DESKTOP_DEVICE_ID
            print(f"📋 Found DESKTOP_DEVICE_ID in config: {DESKTOP_DEVICE_ID}")
            device_id = DESKTOP_DEVICE_ID
        except ImportError:
            print("⚠️  Could not import config, using device ID 5 as fallback")
            device_id = 5
        
        return test_device_recording(device_id)
        
    except Exception as e:
        print(f"❌ Error in config test: {e}")
        return False

def interactive_test():
    """Interactive mode to test different devices"""
    input_devices = list_audio_devices()
    
    print(f"\n🎯 Input devices found: {input_devices}")
    
    while True:
        try:
            choice = input(f"\nEnter device ID to test (or 'q' to quit, 'c' for config test): ").strip()
            
            if choice.lower() == 'q':
                break
            elif choice.lower() == 'c':
                test_desktop_audio_config()
            else:
                device_id = int(choice)
                if device_id in input_devices:
                    test_device_recording(device_id)
                else:
                    print(f"❌ Device {device_id} is not available for input")
        except ValueError:
            print("❌ Please enter a valid device ID number")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

def quick_scan():
    """Quickly scan all input devices for activity"""
    input_devices = list_audio_devices()
    
    print(f"\n🔍 Quick scanning all input devices...")
    print("Play some audio on your computer now!")
    
    results = {}
    for device_id in input_devices:
        print(f"\nTesting device {device_id}...")
        status = test_device_recording(device_id, duration=3)
        results[device_id] = status
    
    print(f"\n📋 SCAN RESULTS:")
    print("=" * 40)
    for device_id, status in results.items():
        device_info = sd.query_devices(device_id)
        if status == "good" or status == "loud":
            print(f"✅ Device {device_id}: {device_info['name']} - {status.upper()}")
        elif status == "quiet" or status == "very_quiet":
            print(f"🟡 Device {device_id}: {device_info['name']} - {status.upper()}")
        else:
            print(f"❌ Device {device_id}: {device_info['name']} - {status.upper()}")

def main():
    print("🖥️  Desktop Audio Capture Test")
    print("=" * 50)
    
    print("\nChoose an option:")
    print("1. List all audio devices")
    print("2. Test configured desktop device")
    print("3. Interactive device testing")
    print("4. Quick scan all input devices")
    print("5. Test specific device ID")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            list_audio_devices()
        elif choice == "2":
            test_desktop_audio_config()
        elif choice == "3":
            interactive_test()
        elif choice == "4":
            quick_scan()
        elif choice == "5":
            device_id = int(input("Enter device ID: "))
            test_device_recording(device_id)
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()