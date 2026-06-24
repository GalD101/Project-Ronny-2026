import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# GLOBAL CONFIGURATION & HELPER FUNCTIONS
# ==========================================

# Calculate the next power of 2 for padding (not using this because I am explicitly told to use 2^17)
calculate_target_length = lambda data: 2 ** int(np.ceil(np.log2(len(data))))
FS = 256 # Sampling frequency in Hz

def load_and_pad_eeg(filepath):
    """
    Loads raw EEG data from a tab-separated text file, extracts the C3 and C4 
    electrodes, and pads them to a length of 2^17 points (512 seconds).
    """
    df = pd.read_csv(filepath, sep='\t')
    df.columns = df.columns.str.strip() # just to make sure data is clean
    
    c3_raw = df['C3'].values
    c4_raw = df['C4'].values
    
    target_length = 2**17 
    print(f"  -> Original signal length(c3): {len(c3_raw)} points")
    print(f"  -> Original signal length(c4): {len(c4_raw)} points")
    print(f"  -> Nearest power of 2 for padding(c3): {calculate_target_length(c3_raw)} = 2^{int(np.log2(calculate_target_length(c3_raw)))} points")
    print(f"  -> Nearest power of 2 for padding(c4): {calculate_target_length(c4_raw)} = 2^{int(np.log2(calculate_target_length(c4_raw)))} points")
    print(f"  -> Padding both signals to {target_length} points (512 seconds)... regardless of original length.")

    c3_padded = np.resize(c3_raw, target_length)
    c4_padded = np.resize(c4_raw, target_length)
    
    print(f"  -> Padded signal length: {len(c3_padded)} points (should be 512 * 256 = 131072 = 2^17)")
    
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
    """
    N = len(raw_signal)
    
    pos_start = 7 * (2**9)
    pos_end = 13 * (2**9)
    neg_start = (2**17) - 13 * (2**9)
    neg_end = (2**17) - 7 * (2**9)
    
    band_mask = np.zeros(N, dtype=bool)
    band_mask[pos_start : pos_end + 1] = True
    band_mask[neg_start : neg_end + 1] = True

    filtered_fft = np.fft.fft(raw_signal) * band_mask 
    filtered_signal = np.fft.ifft(filtered_fft)

    return np.real(filtered_signal)

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

    # Step 2: Create the Hilbert transform multiplier (multiply by -i for positive frequencies, i for negative frequencies)
    N = len(signal)
    hilbert_multiplier = np.zeros(N, dtype=complex)
    hilbert_multiplier[1:N//2] = -1j
    hilbert_multiplier[N//2+1:] = 1j
    hilbert_transformed_fft = fft_signal * hilbert_multiplier

    # Step 3: Compute the iFFT of the transformed signal
    return np.real(np.fft.ifft(hilbert_transformed_fft))

if __name__ == "__main__":
    # ==========================================
    # DATA PROCESSING PIPELINE
    # ==========================================
    print("Loading and padding EEG data...")
    c3, c4 = load_and_pad_eeg('./WWT1_MC-P05.txt')
    print("Data loading complete!\n")
    
    window_start = 10 * FS
    window_end = 11 * FS
    time_axis = np.arange(window_start, window_end) / FS  
    
    print("Applying bandpass filters (7-13 Hz)...")
    c3_filtered = bandpass_filter_hardcoded_indices(c3)
    c4_filtered = bandpass_filter_hardcoded_indices(c4)
    print("Filtering complete!\n")

    print("Extracting instantaneous phase...")
    c3_hilbert = custom_hilbert_transform(c3_filtered)
    c4_hilbert = custom_hilbert_transform(c4_filtered)

    phase_c3 = np.arctan2(c3_hilbert, c3_filtered)
    phase_c4 = np.arctan2(c4_hilbert, c4_filtered)
    print("Phase extraction complete!\n")

    print("Calculating phase differences and complex exponentials...")
    # Calculate Phase Synchronization Index (PSI) over the whole dataset
    phase_diff_wrapped_full = np.angle(np.exp(1j * (phase_c4 - phase_c3)))
    complex_exp_full = np.exp(1j * phase_diff_wrapped_full)
    psi = np.abs(np.mean(complex_exp_full))
    
    print("SUCCESS!")
    print(f"Phase Synchronization Index for C3-C4 (Alpha Band): {psi:.4f}")

    # ==========================================
    # PUBLICATION-STYLE PLOTTING (FINAL 3 WINDOWS ONLY)
    # ==========================================
    
    # Slice the arrays for the 1-second window
    t_win = time_axis
    c3_raw_win = c3[window_start:window_end]
    c3_filt_win = c3_filtered[window_start:window_end]
    c4_raw_win = c4[window_start:window_end]
    c4_filt_win = c4_filtered[window_start:window_end]
    
    p_c3_win = phase_c3[window_start:window_end]
    p_c4_win = phase_c4[window_start:window_end]
    p_diff_win = phase_diff_wrapped_full[window_start:window_end]
    c_exp_win = complex_exp_full[window_start:window_end] 

    # ---------------------------------------------------------
    # WINDOW 1: Panels (a), (b), and (c) Stacked
    # ---------------------------------------------------------
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(t_win, c3_raw_win, color='black', label='Raw', linewidth=1)
    ax1.plot(t_win, c3_filt_win, color='red', label='Filtered (7-13 Hz)', linewidth=2)
    ax1.set_ylabel("C3 [µV]")
    ax1.set_title("(a) C3 Signal")
    ax1.legend(loc='upper right')

    ax2.plot(t_win, c4_raw_win, color='black', label='Raw', linewidth=1)
    ax2.plot(t_win, c4_filt_win, color='blue', label='Filtered (7-13 Hz)', linewidth=2)
    ax2.set_ylabel("C4 [µV]")
    ax2.set_title("(b) C4 Signal")
    ax2.legend(loc='upper right')

    ax3.plot(t_win, p_c3_win, color='red', label='Φ C3', linewidth=2)
    ax3.plot(t_win, p_c4_win, color='blue', label='Φ C4', linewidth=2)
    ax3.set_ylabel("Phase [rad]")
    ax3.set_title("(c) Phase Angle")
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_yticks([-np.pi, 0, np.pi])
    ax3.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax3.legend(loc='upper right')

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(10, 11)
        ax.grid(False)
    
    fig1.tight_layout()

    # ---------------------------------------------------------
    # WINDOW 2: Panel (d) Phase Difference
    # ---------------------------------------------------------
    fig2, ax4 = plt.subplots(figsize=(10, 4))
    
    ax4.plot(t_win, p_diff_win, marker='o', markerfacecolor='none', 
             markeredgecolor='black', markersize=4, linestyle='None')
    
    idx_blue = int(len(t_win) * 0.3) 
    idx_magenta = int(len(t_win) * 0.8) 
    
    ax4.plot(t_win[idx_blue], p_diff_win[idx_blue], marker='o', 
             color='blue', markersize=10, linestyle='None')
             
    ax4.plot(t_win[idx_magenta], p_diff_win[idx_magenta], marker='o', 
             color='magenta', markersize=10, linestyle='None')

    ax4.set_ylabel("ΔΦ [rad]")
    ax4.set_title("(d) Phase Difference (C4 - C3)")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylim(-np.pi, np.pi)
    ax4.set_yticks([-np.pi, 0, np.pi])
    ax4.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax4.set_xlim(10, 11)
    ax4.grid(False)
    fig2.tight_layout()

    # ---------------------------------------------------------
    # WINDOW 3: Panel (e) Complex Exponentials (Unit Circle)
    # ---------------------------------------------------------
    fig3, ax5 = plt.subplots(figsize=(6, 6))
    
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    ax5.add_patch(circle)
    
    ax5.plot(np.real(c_exp_win), np.imag(c_exp_win), marker='o', 
             markerfacecolor='none', markeredgecolor='black', 
             markersize=4, linestyle='None', alpha=0.7, label='exp(i*ΔΦ)')
    
    ax5.plot(np.real(c_exp_win[idx_blue]), np.imag(c_exp_win[idx_blue]), 
             marker='o', color='blue', markersize=10, linestyle='None')
             
    ax5.plot(np.real(c_exp_win[idx_magenta]), np.imag(c_exp_win[idx_magenta]), 
             marker='o', color='magenta', markersize=10, linestyle='None')
    
    mean_vector = np.mean(c_exp_win)
    ax5.arrow(0, 0, np.real(mean_vector), np.imag(mean_vector), 
              color='red', head_width=0.05, length_includes_head=True, linewidth=2)
    ax5.plot(np.real(mean_vector), np.imag(mean_vector), marker='o', 
             color='red', markersize=8, linestyle='None', label='Mean (PSI)') 

    ax5.set_aspect('equal') 
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax5.set_title("(e) Complex Exponentials")
    ax5.set_xlabel("Real Part")
    ax5.set_ylabel("Imaginary Part")
    ax5.legend(loc='upper right', fontsize='small')
    fig3.tight_layout()

    # Show all three windows at once
    plt.show()