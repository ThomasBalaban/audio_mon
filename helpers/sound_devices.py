import sounddevice as sd

def list_input_devices():
    print("\n=== AVAILABLE INPUT DEVICES ===\n")
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            print(f"ID: {index} | {device['name']}")
            print(f"  Channels: {device['max_input_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']} Hz\n")

def list_output_devices():
    print("\n=== AVAILABLE OUTPUT DEVICES ===\n")
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device["max_output_channels"] > 0:
            print(f"ID: {index} | {device['name']}")
            print(f"  Channels: {device['max_output_channels']}")
            print(f"  Default Sample Rate: {device['default_samplerate']} Hz\n")

def list_all_devices():
    print("==============================")
    print("    AUDIO DEVICE LISTING     ")
    print("==============================")
    
    # List input devices
    list_input_devices()
    
    # List output devices
    list_output_devices()
    
    # Show default devices
    try:
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        print("\n=== DEFAULT DEVICES ===\n")
        print(f"Default Input:  ID {default_input['index']} | {default_input['name']}")
        print(f"Default Output: ID {default_output['index']} | {default_output['name']}")
    except Exception as e:
        print(f"Could not determine default devices: {e}")

if __name__ == "__main__":
    try:
        list_all_devices()
    except Exception as e:
        print(f"Error: {e}\nIs sounddevice installed? Run 'pip install sounddevice'")