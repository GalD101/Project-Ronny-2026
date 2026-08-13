import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

def safe_zscore(series: pd.Series, ddof: int = 1) -> pd.Series | None:
    """
    Safely standardizes a Series using (x - mean) / SD.
    Returns None if the standard deviation is 0 or NaN (flatline / zero variance),
    preventing division-by-zero errors.
    """
    std = series.std(ddof=ddof)
    if std == 0 or np.isnan(std):
        return None
    return (series - series.mean()) / std

def align_and_preprocess_subject(
    pid: str,
    hr_dir: str = './hr_1hz',
    sigma_dir: str = './Sigma_Envelope_N2_10_15',
    use_power: bool = True
) -> pd.DataFrame:
    hr_path = os.path.join(hr_dir, f"{pid}.csv")
    sigma_path = os.path.join(sigma_dir, f"{pid}.csv")

    if not os.path.exists(hr_path) or not os.path.exists(sigma_path):
        raise FileNotFoundError(f"Missing input files for subject {pid}.")

    df_hr = pd.read_csv(hr_path)
    df_sigma = pd.read_csv(sigma_path, usecols=['time', 'envelope'])

    if use_power:
        df_sigma['envelope'] = df_sigma['envelope'] ** 2

    mean_n2_envelope = df_sigma['envelope'].mean()
    if mean_n2_envelope == 0 or np.isnan(mean_n2_envelope):
        raise ValueError(f"Sigma envelope mean is 0 or NaN for subject {pid}.")

    df_aligned = pd.merge(df_hr, df_sigma, on='time', how='inner')
    df_aligned = df_aligned.sort_values('time').reset_index(drop=True)

    df_aligned['sigma_pct'] = (100.0 * df_aligned['envelope']) / mean_n2_envelope

    df_aligned['sigma_smooth'] = (
        df_aligned['sigma_pct']
        .rolling(window=4, center=True)
        .mean()
    )

    df_clean = (
        df_aligned[['time', 'hr', 'sigma_pct', 'sigma_smooth']]
        .dropna()
        .reset_index(drop=True)
    )

    return df_clean

def cut_and_zscore_windows(df_clean: pd.DataFrame, window_len: int = 120) -> list[pd.DataFrame]:
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    windows = []
    
    for _, stretch in df_clean.groupby(stretch_ids):
        n_seconds = len(stretch)
        if n_seconds < window_len:
            continue
            
        n_windows = n_seconds // window_len
        stretch_truncated = stretch.iloc[: n_windows * window_len].copy()
        stretch_truncated['window_id'] = np.arange(len(stretch_truncated)) // window_len
        
        for _, win in stretch_truncated.groupby('window_id'):
            win_copy = win.copy()
            hr_z = safe_zscore(win_copy['hr'], ddof=1)
            sigma_z = safe_zscore(win_copy['sigma_smooth'], ddof=1)

            if hr_z is None or sigma_z is None:
                continue
                
            win_copy['hr_z'] = hr_z
            win_copy['sigma_z'] = sigma_z
            windows.append(win_copy[['time', 'hr', 'sigma_smooth', 'hr_z', 'sigma_z']].reset_index(drop=True))
            
    return windows

def bounded_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> np.ndarray:
    n = len(x)
    full_xcorr = np.correlate(x, y, mode='full')
    zero_idx = n - 1
    return full_xcorr[zero_idx - max_lag : zero_idx + max_lag + 1]

def compute_subject_xcorr(windows: list[pd.DataFrame], max_lag: int = 60, min_windows: int = 3) -> pd.DataFrame | None:
    if len(windows) < min_windows:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    window_xcorrs = []
    
    for win in windows:
        hr_z = win['hr_z'].to_numpy()
        sigma_z = win['sigma_z'].to_numpy()
        n = len(hr_z) 

        xcorr_slice = bounded_xcorr(sigma_z, hr_z, max_lag=max_lag)
        overlaps = n - np.abs(lags)
        r_vals = xcorr_slice / overlaps
        window_xcorrs.append(r_vals)

    xcorr_matrix = np.vstack(window_xcorrs)
    
    return pd.DataFrame({
        'lag': lags,
        'r_mean': xcorr_matrix.mean(axis=0),
        'r_std': xcorr_matrix.std(axis=0, ddof=1),
        'n_windows': len(windows)
    })

def compute_grand_average(subject_results: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    valid_subjects = {pid: df for pid, df in subject_results.items() if df is not None}
    n_valid = len(valid_subjects)
    
    if n_valid == 0:
        raise ValueError("No subjects passed the minimum-data rule criteria.")
        
    r_matrix = np.vstack([df['r_mean'].to_numpy() for df in valid_subjects.values()])
    first_df = next(iter(valid_subjects.values())) 
    lags = first_df['lag'].to_numpy()
    
    grand_mean = r_matrix.mean(axis=0)
    sem = r_matrix.std(axis=0, ddof=1) / np.sqrt(n_valid)
    
    return pd.DataFrame({'lag': lags, 'grand_mean': grand_mean, 'sem': sem, 'n_subjects': n_valid})

def compute_subject_psd(df_clean: pd.DataFrame, min_window_sec: int = 180) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Computes Welch's PSD for continuous N2 sleep stretches and averages them per subject.
    """
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    psd_list = []
    freqs = None

    for _, stretch in df_clean.groupby(stretch_ids):
        if len(stretch) >= min_window_sec:
            data = stretch['sigma_smooth'].to_numpy()
            data = data - np.mean(data)

            # Welch's method with a 128s segment to resolve the ~0.02 Hz peak cleanly
            f, pxx = welch(data, fs=1.0, window='hann', nperseg=128, noverlap=64)

            if freqs is None:
                freqs = f
            psd_list.append(pxx)

    if psd_list:
        return freqs, np.mean(psd_list, axis=0)
    return None

def compute_subject_naive_fft(df_clean: pd.DataFrame, window_sec: int = 180) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Computes naive FFT power spectrum for continuous N2 sleep stretches
    truncated exactly to `window_sec`, and averages them per subject.
    Kept for mathematical baseline verification and documentation.
    """
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    fft_list = []
    freqs = None

    for _, stretch in df_clean.groupby(stretch_ids):
        if len(stretch) >= window_sec:
            # 1. Truncate to exact length so all FFTs have identical frequency bins
            data = stretch['sigma_smooth'].to_numpy()[:window_sec]

            # 2. Subtract the mean
            data = data - np.mean(data)

            # 3. Naive FFT (power spectrum)
            fft_power = np.abs(np.fft.rfft(data)) ** 2

            # 4. Calculate the frequency X-axis
            if freqs is None:
                freqs = np.fft.rfftfreq(window_sec, d=1.0)

            fft_list.append(fft_power)

    if fft_list:
        return freqs, np.mean(fft_list, axis=0)
    return None

def compute_grand_average_psd(subject_psds: dict[str, tuple[np.ndarray, np.ndarray] | None]) -> pd.DataFrame:
    """
    Stacks all valid subject PSDs and calculates the grand mean across the cohort.
    """
    valid_psds = [psd for psd in subject_psds.values() if psd is not None]
    
    if not valid_psds:
        raise ValueError("No subjects had valid stretches for PSD.")
    
    freqs = valid_psds[0][0] 
    psd_matrix = np.vstack([psd[1] for psd in valid_psds])
    
    grand_mean_psd = np.mean(psd_matrix, axis=0)
    sem_psd = np.std(psd_matrix, axis=0, ddof=1) / np.sqrt(len(valid_psds))
    
    return pd.DataFrame({
        'frequency': freqs,
        'grand_mean': grand_mean_psd,
        'sem': sem_psd,
        'n_subjects': len(valid_psds)
    })

def plot_panel_c_example(
    pid: str, 
    df_clean: pd.DataFrame, 
    duration_sec: int = 180,
    save_filename: str = "Panel_C_Example_Bout.png"
) -> None:
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    selected_stretch = None
    
    for _, stretch in df_clean.groupby(stretch_ids):
        if len(stretch) >= duration_sec:
            mid_idx = len(stretch) // 2
            start_idx = max(0, mid_idx - (duration_sec // 2))
            selected_stretch = stretch.iloc[start_idx : start_idx + duration_sec].copy()
            break
            
    if selected_stretch is None:
        raise ValueError(f"No continuous N2 stretch >= {duration_sec}s found for subject {pid}.")

    t_rel = np.arange(len(selected_stretch))
    sigma_vals = selected_stretch['sigma_smooth'].to_numpy()
    hr_vals = selected_stretch['hr'].to_numpy()

    # --- LOW PASS FILTER (Cutoff ~0.04 Hz) ---
    b, a = butter(2, 0.04 / 0.5, btype='low') 
    sigma_lowpass = filtfilt(b, a, sigma_vals)
    hr_lowpass = filtfilt(b, a, hr_vals)

    fig, ax_left = plt.subplots(figsize=(10, 4.5))

    color_sigma = '#d62728' 
    ax_left.set_xlabel("Time (s)", fontsize=11)
    ax_left.set_ylabel("Sigma-Band Power (% of N2 mean)", color=color_sigma, fontsize=11, fontweight='bold')
    
    # Plot fast raw data transparently, thick lowpass line on top
    ax_left.plot(t_rel, sigma_vals, color=color_sigma, lw=1.0, alpha=0.3)
    line1 = ax_left.plot(t_rel, sigma_lowpass, color=color_sigma, lw=2.5, label="Sigma Power (Low-Pass 0.04 Hz)")
    ax_left.tick_params(axis='y', labelcolor=color_sigma)
    ax_left.grid(True, alpha=0.25)

    ax_right = ax_left.twinx()
    color_hr = '#000000'
    ax_right.set_ylabel("Heart Rate (bpm)", color=color_hr, fontsize=11, fontweight='bold')
    
    # Plot fast raw data transparently, thick lowpass line on top
    ax_right.plot(t_rel, hr_vals, color=color_hr, lw=1.0, alpha=0.3)
    line2 = ax_right.plot(t_rel, hr_lowpass, color=color_hr, lw=2.5, linestyle='-', label="Heart Rate (Low-Pass 0.04 Hz)")
    ax_right.tick_params(axis='y', labelcolor=color_hr)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper right', framealpha=0.9)

    plt.title(f"Panel C — Human N2 Sleep Example Bout (Subject {pid})", fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    print(f"Saved Panel C example plot as '{save_filename}'.")
    plt.show()

if __name__ == "__main__":
    # =========================================================================
    # EXECUTION TOGGLES: Set to True/False to control what the script runs
    # =========================================================================
    RUN_PANEL_C     = True      # Plot single-subject example bout (Panel C)
    RUN_PANEL_D     = True      # Plot single-subject correlogram (Panel D)
    RUN_FULL_COHORT = True      # Run batch processing (Plots Grand Avg Xcorr & Grand Avg PSD)
    
    SAMPLE_SUBJECT_ID = "200006" 
    # =========================================================================

    if RUN_PANEL_C:
        print(f"--- Generating Figure 6 Panel C for Subject {SAMPLE_SUBJECT_ID} ---")
        try:
            df_step2 = align_and_preprocess_subject(SAMPLE_SUBJECT_ID, sigma_dir='./Sigma_Envelope_N2_10_15', use_power=True)
            plot_panel_c_example(SAMPLE_SUBJECT_ID, df_clean=df_step2, duration_sec=180, save_filename=f"Panel_C_Example_Bout_{SAMPLE_SUBJECT_ID}_10_15.png")
        except Exception as e:
            print(f"[ERROR] Could not generate Panel C: {e}")

    if RUN_PANEL_D:
        print(f"\n--- Generating Figure 6 Panel D for Subject {SAMPLE_SUBJECT_ID} ---")
        try:
            df_step2 = align_and_preprocess_subject(SAMPLE_SUBJECT_ID, sigma_dir='./Sigma_Envelope_N2_10_15', use_power=True)
            windows = cut_and_zscore_windows(df_step2, window_len=120)
            df_xcorr = compute_subject_xcorr(windows, max_lag=60, min_windows=3)
            
            if df_xcorr is not None:
                plt.figure(figsize=(8, 4))
                plt.plot(df_xcorr['lag'], df_xcorr['r_mean'], color='black', lw=1.5, label=f'Subject {SAMPLE_SUBJECT_ID}')
                plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Lag 0')
                plt.title(f"Panel D — Single-Subject Cross-Correlogram ({SAMPLE_SUBJECT_ID})")
                plt.xlabel("Lag τ (s) [Positive = Sigma lags HR]")
                plt.ylabel("Correlation (r)")
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(f"Panel_D_Single_Subject_{SAMPLE_SUBJECT_ID}_10_15.png", dpi=300)
                plt.show()
        except Exception as e:
            print(f"[ERROR] Could not generate Panel D: {e}")

    if RUN_FULL_COHORT:
        print("\n--- Starting Full Cohort Processing (Xcorr & PSD) ---")
        hr_files = sorted(glob.glob('./hr_1hz/*.csv'))
        all_pids = [os.path.splitext(os.path.basename(f))[0] for f in hr_files]
        total_subjects = len(all_pids)
        
        subject_results = {}
        subject_psds = {}
        
        for idx, pid in enumerate(all_pids, start=1):
            try:
                df_step2 = align_and_preprocess_subject(pid, sigma_dir='./Sigma_Envelope_N2_10_15', use_power=True)
                
                # 1. Subject-level Cross Correlation
                windows = cut_and_zscore_windows(df_step2, window_len=120)
                df_xcorr = compute_subject_xcorr(windows, max_lag=60, min_windows=3)
                subject_results[pid] = df_xcorr
                
                # 2. Subject-level PSD (FFT) - USING WELCH'S METHOD
                psd_data = compute_subject_psd(df_step2, min_window_sec=180)
                
                # (Optional) If you wanted to run the naive FFT, you would swap it here:
                # psd_data = compute_subject_naive_fft(df_step2, window_sec=180)
                
                subject_psds[pid] = psd_data
                
                if df_xcorr is not None:
                    n_win = df_xcorr['n_windows'].iloc[0]
                    print(f"[{idx:4d}/{total_subjects}] [PASS] Subject {pid}: {n_win:3d} windows analyzed.")
                else:
                    print(f"[{idx:4d}/{total_subjects}] [DROP] Subject {pid}: Insufficient windows.")
                    
            except Exception as e:
                print(f"[{idx:4d}/{total_subjects}] [ERROR] Subject {pid} failed: {e}")
                subject_results[pid] = None
                subject_psds[pid] = None

        print("\n--- Cohort Processing Complete. Generating Plots ---")
        try:
            # --- PLOT 1: Grand Average Cross-Correlation ---
            df_panel_f = compute_grand_average(subject_results)
            peak_row = df_panel_f.loc[df_panel_f['grand_mean'].idxmax()]
            print(f"\nXcorr Group-Level Peak: r = {peak_row['grand_mean']:.4f} at lag = {int(peak_row['lag'])} s")
            
            lags, g_mean, g_sem = df_panel_f['lag'], df_panel_f['grand_mean'], df_panel_f['sem']
            
            plt.figure(figsize=(9, 5))
            plt.plot(lags, g_mean, color='#1f77b4', lw=2.0, label='Grand Mean (N2 Sleep)')
            plt.fill_between(lags, g_mean - g_sem, g_mean + g_sem, color='#1f77b4', alpha=0.25, label=f'± 1 SEM (N = {df_panel_f["n_subjects"].iloc[0]})')
            plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Lag 0')
            plt.axhline(0, color='black', linestyle='-', alpha=0.3)
            plt.title("Panel F — Grand-Mean Heart Rate / Sigma-Band Power Cross-Correlation")
            plt.xlabel("Lag τ (s) [Positive = Sigma lags HR]")
            plt.ylabel("Correlation (r)")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig("Panel_F_Grand_Average_Reproduction_10_15.png", dpi=300)
            plt.show()

            # --- PLOT 2: Grand Average PSD (FFT) ---
            df_psd_avg = compute_grand_average_psd(subject_psds)
            freqs, psd_mean, psd_sem = df_psd_avg['frequency'], df_psd_avg['grand_mean'], df_psd_avg['sem']
            
            plt.figure(figsize=(9, 5))
            plt.plot(freqs, psd_mean, color='#d62728', lw=2.5, label='Grand Mean PSD')
            plt.fill_between(freqs, psd_mean - psd_sem, psd_mean + psd_sem, color='#d62728', alpha=0.25, label=f'± 1 SEM (N = {df_psd_avg["n_subjects"].iloc[0]})')
            plt.axvline(0.02, color='black', linestyle='--', alpha=0.7, label='0.02 Hz (~50s cycle)')
            plt.xlim(0, 0.1) # Zoom into infraslow rhythm
            plt.title("Grand Average PSD of Sigma Power (10-15 Hz)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig("Grand_Average_FFT_10_15.png", dpi=300)
            plt.show()
            
        except Exception as e:
            print(f"[FATAL ERROR] Could not compute grand averages: {e}")