import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# GLOBAL CONFIGURATION & HELPER FUNCTIONS
# ==========================================

# DON'T USE THIS! FOR SOME REASON WE NEED TO PAD TO 2**17 AND NOT 2**16 (which is the closest next power of two).
calculate_target_length = lambda data: 2 ** int(np.ceil(np.log2(len(data))))
FS = 256 # Sampling frequency in Hz

def load_and_pad_eeg(filepath):
    """
    Loads raw EEG data from a tab-separated text file, extracts the C3 and C4 
    electrodes, and pads them to a length of 2^17 points (512 seconds) by looping.
    """
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip() # just to make sure data is clean
    
    # Extract the C3 and C4 leads as NumPy arrays
    c3_raw = df['C3'].values
    c4_raw = df['C4'].values
    
    print(f"  -> Original signal length: {len(c3_raw)} points")

    target_length = 2**17 # calculate_target_length(c3_raw)
    
    # Pad to exactly 2^17 points by copy-pasting additional copies of the time series as instructed. (I think it is FFT friendly but I need to check why 2**17 and not 2**16 (which is the closest power of 2)? Maybe we want a longer sequence?)
    # np.resize automatically loops the array if the target is larger
    c3_padded = np.resize(c3_raw, target_length)
    c4_padded = np.resize(c4_raw, target_length)
    
    print(f"  -> Padded signal length: {len(c3_padded)} points (should be 512 * 256 = 131072)")
    
    return c3_padded, c4_padded

def bandpass_filter_dynamic(raw_signal, low_freq, high_freq, fs):
    """
    Filters a signal in the frequency domain using dynamically calculated physical Hz bins.
    Handles any array size or sampling frequency to isolate the specific frequency band.
    """

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
    # np.abs() dynamically handles both the positive and negative frequency blocks.
    band_mask = (np.abs(freq_bins) >= low_freq) & (np.abs(freq_bins) <= high_freq)

    # Apply the mask to the FFT coefficients
    # Multiplying by the boolean mask explicitly sets all other complex coefficients to zero.
    # * = (element-wise multiplication) (e.g. [1, 2, 3] * [True, False, True] = [1, 0, 3])
    filtered_fft = np.fft.fft(raw_signal) * band_mask # multiplication in frequency domain is convolution in time domain

    # Inverse FFT to get the filtered signal back in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)

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

def bandpass_filter_hardcoded_indices(raw_signal):
    """
    Strict filter using the suggested hardcoded mathematical indices for the Alpha band.
    This assumes the input signal length is EXACTLY 2^17 and FS is 256 Hz.
    """
    N = len(raw_signal) # Assumed to be 2^17 = 131072
    
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
    # Multiplying by the boolean mask explicitly sets all other complex coefficients to zero.
    filtered_fft = np.fft.fft(raw_signal) * band_mask # multiplication in frequency domain is convolution in time domain

    # 5. Inverse FFT to get the filtered signal back in the time domain
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)

def plot_raw_vs_filtered(time_axis, raw_signal, filtered_signal, title, filtered_color):
    """
    Generates a localized plot comparing the raw voltage to the filtered oscillatory 
    wave over a specific 1-second window to verify filter integrity.
    """
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

def custom_hilbert_transform(signal):
    """
    Computes the Hilbert transform (the 90-degree phase shift) of a real signal.
    """

    # Hilbert transform:
    # 1. Take FFT of signal
    # 2. Rotate Fourier coefficients by 90 degrees (cos -> sin, sin -> -cos) (same as multiplying by i in the frequency domain)
    # 3. Take iFFT of rotated Fourier coefficients
    # Used information from this video: https://youtu.be/VyLU8hlhI-I

    # Step 1: Compute the FFT of the signal
    fft_signal = np.fft.fft(signal)

    # Step 2: Multiply by the Hilbert transform multiplier 
    # (Multiply by -i for positive frequencies, +i for negative frequencies)
    N = len(signal)
    hilbert_multiplier = np.zeros(N, dtype=complex)
    hilbert_multiplier[1:N//2] = -1j  # Positive frequencies
    hilbert_multiplier[N//2+1:] = 1j  # Negative frequencies
    
    hilbert_transformed_fft = fft_signal * hilbert_multiplier

    # Step 3: Take the inverse FFT to get the Hilbert transform in the time domain
    # We use np.real() to discard small imaginary rounding errors
    return np.real(np.fft.ifft(hilbert_transformed_fft))


if __name__ == "__main__":
    # ==========================================
    # DATA LOADING & PADDING
    # ==========================================
    print("Loading and padding EEG data...")
    c3, c4 = load_and_pad_eeg('./WWT1_MC-P05.txt')
    print("Data loading complete!\n")
    
    # Graph should show data from 10th to 11th second (like in fig5k.pdf)
    window_start = 10 * FS
    window_end = 11 * FS
    time_axis = np.arange(window_start, window_end) / FS  # 10th to 11th second
    
    # ==========================================
    # FREQUENCY FILTERING (ALPHA BAND 7-13 Hz)
    # ==========================================
    print("Applying bandpass filters (7-13 Hz)...")
    low_freq = 7
    high_freq = 13
    
    # c3_filtered = bandpass_filter_dynamic(c3, low_freq, high_freq, FS)
    # c4_filtered = bandpass_filter_dynamic(c4, low_freq, high_freq, FS)
    # Using the hardcoded version, but dynamic version is available
    c3_filtered = bandpass_filter_hardcoded_indices(c3)
    c4_filtered = bandpass_filter_hardcoded_indices(c4)
    print("Filtering complete!\n")

    # ==========================================
    # VERIFICATION PLOTS
    # ==========================================
    print("Plotting raw vs filtered signals...")
    
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
    print("Plots generated!\n")

    # ==========================================
    # HILBERT TRANSFORM & PHASE EXTRACTION
    # ==========================================
    print("Extracting instantaneous phase...")
    
    # 1. Apply the manual Hilbert transform to the ALREADY FILTERED signals
    c3_hilbert = custom_hilbert_transform(c3_filtered)
    c4_hilbert = custom_hilbert_transform(c4_filtered)

    # 2. Extract the instantaneous phase using arctan2
    # According to the instructions: atan2(Hilbert Transformed, Real Filtered)
    phase_c3 = np.arctan2(c3_hilbert, c3_filtered)
    phase_c4 = np.arctan2(c4_hilbert, c4_filtered)

    print("Phase extraction complete!\n")

    # ==========================================
    # PHASE DIFFERENCES & SYNCHRONIZATION
    # ==========================================
    print("Calculating phase differences and complex exponentials...")
    
    # (d) Calculate phase differences (ΔΦ)
    phase_diff = phase_c3 - phase_c4
    
    # Plot Phase Difference (d) for the 10th to 11th second
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, phase_diff[window_start:window_end], color='purple', label='Δ Phase (C3 - C4)')
    plt.title("Phase Difference (ΔΦ) over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Phase Difference [rad]")
    plt.xlim(10, 11)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (e) Calculate complex exponentials of the phase differences
    complex_exp = np.exp(1j * phase_diff)
    
    # Plot the Real part of the Complex Exponential (e)
    # The real part is mathematically equivalent to cos(ΔΦ)
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, np.real(complex_exp)[window_start:window_end], color='green', label='Re{ exp(i*ΔΦ) }')
    plt.title("Complex Exponential of Phase Difference (Real Part)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.xlim(10, 11)
    plt.ylim(-1.5, 1.5)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate the final Phase Synchronization Index (PSI)
    # PSI = | < e^(i * ΔΦ) > | (The absolute value of the mean vector)
    psi = np.abs(np.mean(complex_exp))
    
    print("SUCCESS!")
    print(f"Phase Synchronization Index for C3-C4 (Alpha Band): {psi:.4f}")