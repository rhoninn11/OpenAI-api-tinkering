import os
import sys
import time
import scipy
import numpy as np
import sounddevice as sd
from pathlib import Path
from datetime import datetime

# Create the output directory if it doesn't exist
output_folder = Path("fs")
output_folder.mkdir(parents=True, exist_ok=True)

# Change this to the desired file format (e.g. 'wav', 'flac', etc.)
FILE_FORMAT = 'wav'

# Buffer to store recorded data
buffer = []

# Callback function for recording audio
def audio_callback(indata, frames, time, status):
    buffer.append(indata.copy())

def check_device_direction(device):
    device_type = []
    if device['max_input_channels'] > 0:
        device_type.append('Input')
    if device['max_output_channels'] > 0:
        device_type.append('Output')
    return ' & '.join(device_type)

def device_select():
    devices = sd.query_devices()
    print("Available recording devices:")
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} ({check_device_direction(device)})")

    device_index = int(input("Enter the index of the device you want to use for recording: "))
    return device_index

def get_device_info(device_index):
    device_info = sd.query_devices(device_index, 'input')
    sample_rate = int(device_info['default_samplerate'])
    channels = device_info['max_input_channels']
    return sample_rate, channels

# Recording settings
device_info = sd.query_devices(None, 'input')
channels = device_info['max_input_channels']
dtype = 'float32'

try:
    device_idx = device_select()
    SAMPLE_RATE, channels = get_device_info(device_idx)

    stream_config = {
        'device': device_idx,
        'channels': channels,
        'samplerate': SAMPLE_RATE,
        'callback': audio_callback,
    }
    print(f"Starting recording dev({device_idx}: {SAMPLE_RATE}x{channels}). Press Ctrl+C to stop.")
    with sd.InputStream(**stream_config):
        while True:
            time.sleep(0.05)

except KeyboardInterrupt:
    # Save the recorded data to a file
    print("Saving recorded audio...")
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_folder / f'recorded_{file_timestamp}.{FILE_FORMAT}'
    print(f"+++ Saving to {output_file}...")


    recorded_data = np.concatenate(buffer, axis=0)
    rcd_shape = recorded_data.shape
    rcd_dtype = recorded_data.dtype
    print(f"+++ Recorded data shape: {rcd_shape}")
    print(f"+++ Recorded data dtype: {rcd_dtype}")

    # float32 to int16
    recorded_data = recorded_data * 32767
    recorded_data = recorded_data.astype(np.int16)

    # max value
    max_value = np.max(recorded_data)

    # min value
    min_value = np.min(recorded_data)

    print(f"+++ Max value: {max_value}")
    print(f"+++ Min value: {min_value}")

    scipy.io.wavfile.write(str(output_file), SAMPLE_RATE, recorded_data)
    # sd.write_wav(str(output_file), recorded_data, SAMPLE_RATE)
    
    print(f"Audio saved to {output_file}")
    sys.exit(0)