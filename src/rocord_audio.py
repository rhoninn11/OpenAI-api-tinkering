import os, json
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

def dev_id_string(device):
    return f"{device['name']} ({check_device_direction(device)})"

def save_device_config(device_index):
    config_folder = Path('fs/config')
    config_folder.mkdir(parents=True, exist_ok=True)
    config_file = config_folder / 'config.json'

    devices = sd.query_devices()
    device = devices[device_index]
    device_config = {'idx': device_index, 'id_string': dev_id_string(device)}

    with open(config_file, 'w') as f:
        json.dump(device_config, f)

def load_device_config():
    config_folder = Path('fs/config')
    config_file = config_folder / 'config.json'

    if config_file.exists():
        with open(config_file, 'r') as f:
            device_config = json.load(f)
        return device_config
    else:
        return None

def user_select_input_device():
    devices = sd.query_devices()
    print("Available recording devices:")
    for idx, device in enumerate(devices):
        id_string = dev_id_string(device)
        print(f"{idx}: {id_string}")

    device_index = int(input("Enter the index of the device you want to use for recording: "))
    return device_index

def config_select_input_device():
    config = load_device_config()
    if config is None:
        return None
        
    devices = sd.query_devices()
    idx_config = config['idx']
    id_string_config = config['id_string']

    print(f"+++ config: {idx_config} - {id_string_config}")
    if idx_config < len(devices):
        device_config = devices[idx_config]
        loaded_id_string = dev_id_string(device_config)
        print(f"+++ loaded: {idx_config} - {loaded_id_string}")
        if id_string_config == loaded_id_string:
            print(f"+++ using previous device")
            return idx_config
        
    return None

def device_select():
    dev_idx = config_select_input_device()
    if dev_idx is not None:
        print(f"+++ device selected form config: {dev_idx}")
        return dev_idx

    dev_idx = user_select_input_device()
    save_device_config(dev_idx)
    print(f"+++ device selected form user: {dev_idx}")
    print(f"+++ config saved:D")
    return dev_idx

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

    WHISPER_SAMPLERATE = 16000
    # decimated_data = scipy.signal.decimate(recorded_data, 3)
    # dec_shape = decimated_data.shape
    # print(f"+++ Decimated data shape: {dec_shape}")

    simple_downsample = recorded_data[::3,:]
    scipy.io.wavfile.write(str(output_file), WHISPER_SAMPLERATE, simple_downsample)
    # sd.write_wav(str(output_file), recorded_data, SAMPLE_RATE)
    
    print(f"Audio saved to {output_file}")
    sys.exit(0)