import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_LENGTH = 2**17  # Target length for fft (elongate the data by padding to this length) (512 seconds at 256 Hz)
FS = 256               # Sampling frequency in Hz

def load_and_pad_eeg(filepath, target_length):
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip() # just to make sure data is clean
    
    # Extract the C3 and C4 leads as NumPy arrays
    c3_raw = df['C3'].values
    c4_raw = df['C4'].values
    
    print(f"Original signal length: {len(c3_raw)} points")
    
    # Pad to exactly 2^17 points by repeating the array (fft friendly)
    # np.resize automatically loops the array if the target is larger
    c3_padded = np.resize(c3_raw, target_length)
    c4_padded = np.resize(c4_raw, target_length)
    
    print(f"Padded signal length: {len(c3_padded)} points (should be 512 * 256 = 131072)")
    
    return c3_padded, c4_padded

def bandpass_filter(raw_signal, low_freq, high_freq, fs):
    # Apply FFT to the padded signals
    # Note: Here is the pseudo code for FFT (if I will implement it myself, I will get rounding errors that will accunulate from the "twiddle factors")
    # FFT (arr, n):
    #     if n == 1:
    #         return arr[0]
    #     even = FFT(arr[0::2], n//2)
    #     odd = FFT(arr[1::2], n//2)
    #     P_even = FFT(even, n//2)
    #     P_odd = FFT(odd, n//2)
    #     y = [0] * n
    #     for j in range(n//2):
    #         w = exp(-2 * pi * j / n)  # Twiddle factor
    #         y[j] = P_even[j] + w * P_odd[j]
    #         y[j + n//2] = P_even[j] - w * P_odd[j]
    #    return y

    # Compute the frequency bins for the FFT
    freq_bins = np.fft.fftfreq(len(raw_signal), d=1/fs)

    # Create a mask to keep only the frequencies in the desired band
    band_mask = (freq_bins >= low_freq) & (freq_bins <= high_freq)

    # Apply the mask to the FFT coefficients
    filtered_fft = np.fft.fft(raw_signal) * band_mask

    # Inverse FFT to get the filtered signal back in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)


if __name__ == "__main__":
    c3, c4 = load_and_pad_eeg('./WWT1_MC-P05.txt', TARGET_LENGTH)
    
    time_axis = np.arange(1000) / FS  # First 1000 points (1,000 / 256 ~3.9 seconds)
    
    # plt.figure(figsize=(10, 4))
    # plt.plot(time_axis, c3[:1000], label='C3 Lead', color='red')
    # plt.plot(time_axis, c4[:1000], label='C4 Lead', color='blue', alpha=0.7)
    # plt.title("Raw EEG Data (First ~4 seconds)")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Voltage")
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # Apply band pass filter (7-13 Hz)
    low_freq = 7
    high_freq = 13
    c3_filtered = bandpass_filter(c3, low_freq, high_freq, FS)
    c4_filtered = bandpass_filter(c4, low_freq, high_freq, FS)

    # plot C3 raw vs filtered
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, c3[:1000], label='C3 Raw', color='black')
    plt.plot(time_axis, c3_filtered[:1000], label='C3 Filtered (7-13 Hz)', color='red')
    plt.title("C3 Signal: Raw vs Bandpass Filtered")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot C4 raw vs filtered
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, c4[:1000], label='C4 Raw', color='black')
    plt.plot(time_axis, c4_filtered[:1000], label='C4 Filtered (7-13 Hz)', color='blue')
    plt.title("C4 Signal: Raw vs Bandpass Filtered")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()