import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# DON'T USE THIS! FOR SOME REASON WE NEED TO PAD TO 2**17 AND NOT 2**16.
calculate_target_length = lambda data: 2 ** int(np.ceil(np.log2(len(data))))
FS = 256 # Sampling frequency in Hz

def load_and_pad_eeg(filepath):
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip() # just to make sure data is clean
    
    # Extract the C3 and C4 leads as NumPy arrays
    c3_raw = df['C3'].values
    c4_raw = df['C4'].values
    
    print(f"Original signal length: {len(c3_raw)} points")

    target_length = 2**17 # calculate_target_length(c3_raw)
    
    # Pad to exactly 2^17 points by copy-pasting additional copies of the time series as instructed. (I think it is FFT friendly but I need to check why 2**17 and not 2**16? Maybe we want a longer sequence?)
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

    # --- NOTE ON FREQUENCY INDICES ---
    # The instructions suggested using hardcoded indices (7*2^9 to 13*2^9 and the 
    # corresponding negative frequencies 2^17 - 13*2^9 to 2^17 - 7*2^9).
    # Instead, this code dynamically calculates the physical Hz bins using np.fft.fftfreq.
    # Because our resolution is exactly fs/N = 256 / 2^17 = 1/512 Hz per step, 
    # mapping 7 Hz dynamically is mathematically identical to targeting index 7 * 2^9.
    # Using np.abs() dynamically handles both the positive and negative frequency blocks.

    # Compute the frequency bins for the FFT
    freq_bins = np.fft.fftfreq(len(raw_signal), d=1/fs)

    # Create a mask to keep only the frequencies in the desired band
    band_mask = (np.abs(freq_bins) >= low_freq) & (np.abs(freq_bins) <= high_freq)

    # Apply the mask to the FFT coefficients
    # [1, 2, 3] * [4, 5, 6] = [4, 10, 18] (element-wise multiplication)
    filtered_fft = np.fft.fft(raw_signal) * band_mask # multiplication in frequency domain is convolution in time domain

    # Inverse FFT to get the filtered signal back in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)


def bandpass_filter_hardcoded_indices(raw_signal):
    """
    Strict filter using the suggested hardcoded mathematical indices for the Alpha band.
    This assumes the input signal length is EXACTLY 2^17 and FS is 256 Hz.
    """
    N = len(raw_signal) # Assumed to be 131072
    
    # 1. Define the suggested exact indices
    pos_start = 7 * (2**9)
    pos_end = 13 * (2**9)
    
    neg_start = (2**17) - 13 * (2**9)
    neg_end = (2**17) - 7 * (2**9)
    
    # 2. Initialize an all-zero (False) logical mask
    band_mask = np.zeros(N, dtype=bool)
    
    # 3. Set the specific index windows to True
    # We add +1 to the end indices because Python slices (start:end) exclude the last number
    band_mask[pos_start : pos_end + 1] = True
    band_mask[neg_start : neg_end + 1] = True

    # 4. Apply the mask to the FFT coefficients
    # [1, 2, 3] * [4, 5, 6] = [4, 10, 18] (element-wise multiplication)
    filtered_fft = np.fft.fft(raw_signal) * band_mask # multiplication in frequency domain is convolution in time domain

    # 5. Inverse FFT to get the filtered signal back in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)

def plot_raw_vs_filtered(time_axis, raw_signal, filtered_signal, title, filtered_color):
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, raw_signal, label='Raw', color='black')
    plt.plot(time_axis, filtered_signal, label='Filtered (7-13 Hz)', color=filtered_color)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [µV]")
    plt.xlim(10, 11)
    plt.ylim(-20, 20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and pad the EEG data
    c3, c4 = load_and_pad_eeg('./WWT1_MC-P05.txt')
    
    # Graph should show data from 10th to 11th second (like in fig5k.pdf)
    window_start = 10 * FS
    window_end = 11 * FS
    time_axis = np.arange(window_start, window_end) / FS  # 10th to 11th second
    
    # Apply band pass filter (7-13 Hz)
    low_freq = 7
    high_freq = 13
    c3_filtered = bandpass_filter(c3, low_freq, high_freq, FS)
    c4_filtered = bandpass_filter(c4, low_freq, high_freq, FS)

    # plot C3 raw vs filtered
    plot_raw_vs_filtered(
        time_axis,
        c3[window_start:window_end],
        c3_filtered[window_start:window_end],
        "C3 Signal: Raw vs Bandpass Filtered",
        "red",
    )

    # plot C4 raw vs filtered
    plot_raw_vs_filtered(
        time_axis,
        c4[window_start:window_end],
        c4_filtered[window_start:window_end],
        "C4 Signal: Raw vs Bandpass Filtered",
        "blue",
    )