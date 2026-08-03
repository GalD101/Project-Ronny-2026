import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    sigma_dir: str = './Sigma_Envelope_N2',
    use_power: bool = True
) -> pd.DataFrame:
    """
    Step 2: Aligns the 1-Hz heart rate trace with the N2 Sigma EEG envelope.
    
    1. Loads the precomputed 1-Hz HR trace from disk.
    2. Loads N2 sigma envelope data.
    3. Performs an inner join on integer 'time' (restricts HR exclusively to N2 sleep).
    4. Normalizes sigma envelope to % of its nightly N2 mean.
    5. Applies a 4-second centered symmetric moving average to sigma_pct.
    6. Leaves instantaneous heart rate unsmoothed.

    Parameters
    ----------
    pid : str
        Subject ID.
    hr_dir : str
        Directory containing precomputed 1-Hz HR traces.
    sigma_dir : str
        Directory containing Sigma N2 CSV files.
    use_power : bool, default True
        If True, squares the amplitude envelope to convert to sigma power before 
        normalizing. If False, uses the raw amplitude envelope.
    """
    hr_path = os.path.join(hr_dir, f"{pid}.csv")
    sigma_path = os.path.join(sigma_dir, f"{pid}.csv")

    if not os.path.exists(hr_path) or not os.path.exists(sigma_path):
        raise FileNotFoundError(f"Missing input files for subject {pid}.")

    # 1. Load both traces
    df_hr = pd.read_csv(hr_path)
    df_sigma = pd.read_csv(sigma_path, usecols=['time', 'envelope'])

    # Square the envelope before normalizing if strict power units are requested
    # OPTIONAL
    if use_power:
        df_sigma['envelope'] = df_sigma['envelope'] ** 2

    # 2. Calculate the nightly N2 mean BEFORE any dropping/joining
    mean_n2_envelope = df_sigma['envelope'].mean()
    if mean_n2_envelope == 0 or np.isnan(mean_n2_envelope):
        raise ValueError(f"Sigma envelope mean is 0 or NaN for subject {pid}.")

    # 3. Inner join on integer 'time' -> restricts data exclusively to shared N2 sleep seconds
    df_aligned = pd.merge(df_hr, df_sigma, on='time', how='inner')
    df_aligned = df_aligned.sort_values('time').reset_index(drop=True)

    # 4. Normalize sigma to % of nightly N2 mean
    df_aligned['sigma_pct'] = (100.0 * df_aligned['envelope']) / mean_n2_envelope

    # 5. Apply a 4-second centered (symmetric) moving average to sigma_pct
    df_aligned['sigma_smooth'] = (
        df_aligned['sigma_pct']
        .rolling(window=4, center=True)
        .mean()
    )

    # 6. Keep clean columns (dropping boundary NaNs from the 4-s rolling window)
    df_clean = (
        df_aligned[['time', 'hr', 'sigma_pct', 'sigma_smooth']]
        .dropna()
        .reset_index(drop=True)
    )

    return df_clean

def cut_and_zscore_windows(df_clean: pd.DataFrame, window_len: int = 120) -> list[pd.DataFrame]:
    """
    Step 3: Cuts continuous N2 stretches into non-overlapping 120-s windows
    and independently z-scores both 'hr' and 'sigma_smooth' within each window.
    
    Parameters
    ----------
    df_clean : pd.DataFrame
        Aligned and preprocessed DataFrame containing 'time', 'hr', 'sigma_smooth'.
    window_len : int, default 120
        Length of each window in seconds.
        
    Returns
    -------
    list[pd.DataFrame]
        A list of DataFrames, where each DataFrame is exactly 120 rows long
        and contains z-scored columns: 'hr_z' and 'sigma_z'.
    """
    # 1. Identify continuous stretches where time increments by exactly 1 second
    # A jump in time (> 1 second) increments the stretch_id
    # neatly use cumsum of boolean mask to create unique stretch IDs
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    
    windows = []
    
    # 2. Process each continuous stretch independently
    for _, stretch in df_clean.groupby(stretch_ids):
        n_seconds = len(stretch)
        
        # If the stretch is shorter than window_len (e.g. 120 s), drop it
        if n_seconds < window_len:
            continue
            
        # Calculate how many complete 120-s windows fit in this stretch
        n_windows = n_seconds // window_len
        
        # Truncate any trailing remainder that doesn't fit into a full 120-s window
        stretch_truncated = stretch.iloc[: n_windows * window_len].copy()
        
        # Assign a local window number (0, 1, 2, ...) within this stretch
        # neatly use the // operator to assign window IDs: 0, 1, 2, ... , window_len will be 0, then window_len + 1, window_len + 2, ... 2*window_len will get 1, etc.
        stretch_truncated['window_id'] = np.arange(len(stretch_truncated)) // window_len
        
        # 3. Z-score 'hr' and 'sigma_smooth' independently within each 120-s window
        for _, win in stretch_truncated.groupby('window_id'):
            win_copy = win.copy()

            # Standardize (z-score) heart rate and smoothed sigma within the window:
            # (x - mean) / SD for heart rate
            # ddof=1 means we use the sample std (the weird N - 1 I learned about back in Probability & Statistics course with Stas (biased vs. unbiased estimators))
            # ddof=0 would be the population std (divide by N instead of N - 1)
            # I use ddof=1 since this is a sample of the population of all possible heart rate values, not the entire population itself
            hr_z = safe_zscore(win_copy['hr'], ddof=1)
            sigma_z = safe_zscore(win_copy['sigma_smooth'], ddof=1)

            # Skip this window if either signal had zero variance or all NaNs
            if hr_z is None or sigma_z is None:
                continue
                
            win_copy['hr_z'] = hr_z
            win_copy['sigma_z'] = sigma_z
            
            windows.append(win_copy[['time', 'hr', 'sigma_smooth', 'hr_z', 'sigma_z']].reset_index(drop=True))
            
    return windows

def bounded_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int = 60) -> np.ndarray:
    """
    Computes the cross-correlation R(tau) = sum(x(t + tau) * y(t))
    and returns ONLY the symmetric window from -max_lag to +max_lag.
    
    Returns an array of length (2 * max_lag + 1).
    """
    n = len(x)

    full_xcorr = np.correlate(x, y, mode='full')
    
    # Why is Lag 0 at index n - 1?
    # mode='full' returns an array of length 2*n - 1 (covering lags from -(n-1) to +(n-1)).
    # The exact center index where neither signal is shifted (Lag 0, perfect alignment) is:
    #   center_index = (2*n - 1) // 2 = n - 1
    zero_idx = n - 1
    
    return full_xcorr[zero_idx - max_lag : zero_idx + max_lag + 1]

def compute_subject_xcorr(
    windows: list[pd.DataFrame], 
    max_lag: int = 60, 
    min_windows: int = 3
) -> pd.DataFrame | None:
    """
    Step 4: Computes the normalized cross-correlation for each 120-s window
    and averages them within-subject.
    """
    # "Minimum-data rule"
    if len(windows) < min_windows:
        return None

    lags = np.arange(-max_lag, max_lag + 1)
    window_xcorrs = []
    
    for win in windows:
        hr_z = win['hr_z'].to_numpy()
        sigma_z = win['sigma_z'].to_numpy()
        n = len(hr_z) # "Heart rate is the source signal." so we will use its length to determine the number of overlapping samples at each lag

        # Get the 121 symmetric lag points (-60 to +60)
        xcorr_slice = bounded_xcorr(sigma_z, hr_z, max_lag=max_lag)
        
        # Normalize by exact number of overlapping samples at each lag
        overlaps = n - np.abs(lags)
        r_vals = xcorr_slice / overlaps
        
        window_xcorrs.append(r_vals)

    # Stack all window curves (shape: [n_windows, len(lags)])
    xcorr_matrix = np.vstack(window_xcorrs)
    
    # Average across windows (within-subject mean)
    df_subject_r = pd.DataFrame({
        'lag': lags,
        'r_mean': xcorr_matrix.mean(axis=0),
        'r_std': xcorr_matrix.std(axis=0, ddof=1),
        'n_windows': len(windows)
    })
    
    return df_subject_r

def compute_grand_average(subject_results: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """
    Stage 2 Averaging: Computes the grand-mean correlogram and standard error 
    of the mean (SEM) across all valid subjects for Panel F.
    
    Parameters
    ----------
    subject_results : dict[str, pd.DataFrame | None]
        Dictionary mapping subject ID to their Step 4 DataFrame (or None if failed QC).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['lag', 'grand_mean', 'sem', 'n_subjects'].
    """
    # 1. Separate subjects who passed QC from those who failed
    valid_subjects = {pid: df for pid, df in subject_results.items() if df is not None}
    n_total = len(subject_results)
    n_valid = len(valid_subjects)
    
    # Report quality control results as required by the guidelines
    print("--- Stage 2 Quality Control Summary ---")
    print(f"Total subjects evaluated : {n_total}")
    print(f"Subjects passed (≥3 win) : {n_valid} ({(n_valid / n_total) * 100:.1f}%)")
    print(f"Subjects dropped         : {n_total - n_valid}")
    
    if n_valid == 0:
        raise ValueError("No subjects passed the minimum-data rule criteria.")
        
    # 2. Extract the 'r_mean' curve from each valid subject
    # Stack into a matrix of shape [n_valid_subjects, 121]
    r_matrix = np.vstack([df['r_mean'].to_numpy() for df in valid_subjects.values()])
    
    # Grab the shared time lag axis (-60 to +60 s) from any valid subject
    first_df = next(iter(valid_subjects.values())) # use iterator to save memory
    lags = first_df['lag'].to_numpy()
    
    # 3. Calculate Grand Mean across subjects (axis=0)
    grand_mean = r_matrix.mean(axis=0)
    
    # 4. Calculate SEM across subjects: SD / sqrt(N)
    grand_sd = r_matrix.std(axis=0, ddof=1)
    sem = grand_sd / np.sqrt(n_valid)
    
    df_grand = pd.DataFrame({
        'lag': lags,
        'grand_mean': grand_mean,
        'sem': sem,
        'n_subjects': n_valid
    })
    
    return df_grand

def plot_panel_c_example(
    pid: str, 
    df_clean: pd.DataFrame, 
    duration_sec: int = 180,
    save_filename: str = "Panel_C_Example_Bout.png"
) -> None:
    """
    Step 5 (Panel C): Plots a representative ~180-second continuous N2 sleep stretch
    with dual Y-axes: Sigma Power % of mean (red, left) and Heart Rate bpm (black, right).
    """
    # 1. Identify continuous N2 sleep stretches where time increments by exactly 1 s
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    
    selected_stretch = None
    
    # 2. Search for a continuous stretch at least 'duration_sec' long
    for _, stretch in df_clean.groupby(stretch_ids):
        if len(stretch) >= duration_sec:
            # Grab a slice from the middle of the stretch to avoid edge artifacts
            mid_idx = len(stretch) // 2
            start_idx = max(0, mid_idx - (duration_sec // 2))
            selected_stretch = stretch.iloc[start_idx : start_idx + duration_sec].copy()
            break
            
    if selected_stretch is None:
        raise ValueError(f"No continuous N2 stretch >= {duration_sec}s found for subject {pid}.")

    # Make time relative to the start of the plotted window (0 to 180 s)
    t_rel = np.arange(len(selected_stretch))
    sigma_vals = selected_stretch['sigma_smooth'].to_numpy()
    hr_vals = selected_stretch['hr'].to_numpy()

    # 3. Create the dual Y-axis plot
    fig, ax_left = plt.subplots(figsize=(10, 4.5))

    # --- LEFT Y-AXIS: Sigma Power (% of N2 Mean) in RED ---
    color_sigma = '#d62728'  # Strong red
    ax_left.set_xlabel("Time (s)", fontsize=11)
    ax_left.set_ylabel("Sigma-Band Power (% of N2 mean)", color=color_sigma, fontsize=11, fontweight='bold')
    line1 = ax_left.plot(t_rel, sigma_vals, color=color_sigma, lw=2.0, label="Sigma Power (11.5–16 Hz)")
    ax_left.tick_params(axis='y', labelcolor=color_sigma)
    ax_left.grid(True, alpha=0.25)

    # --- RIGHT Y-AXIS: Instantaneous Heart Rate (bpm) in BLACK ---
    ax_right = ax_left.twinx()  # Share the same X-axis
    color_hr = '#000000'        # Black
    ax_right.set_ylabel("Heart Rate (bpm)", color=color_hr, fontsize=11, fontweight='bold')
    line2 = ax_right.plot(t_rel, hr_vals, color=color_hr, lw=1.8, linestyle='-', label="Heart Rate (bpm)")
    ax_right.tick_params(axis='y', labelcolor=color_hr)

    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, loc='upper right', framealpha=0.9)

    plt.title(f"Panel C — Human N2 Sleep Example Bout (Subject {pid})", fontsize=13, pad=10)
    plt.tight_layout()
    
    # Save to disk
    plt.savefig(save_filename, dpi=300)
    print(f"Saved Panel C example plot as '{save_filename}'.")
    plt.show()

if __name__ == "__main__":
    test_pid = "200002"
    
    # Run Step 2 to get the clean N2 trace
    df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
    
    # Generate Panel C
    plot_panel_c_example(test_pid, df_step2, duration_sec=180, save_filename="Panel_C_Example_Bout.png")

# if __name__ == "__main__":
#     import glob
#     import matplotlib.pyplot as plt

#     # 1. Automatically find all subject IDs from the CSV files in your HR directory
#     hr_files = sorted(glob.glob('./hr_1hz/*.csv'))
#     all_pids = [os.path.splitext(os.path.basename(f))[0] for f in hr_files]
    
#     print(f"Found {len(all_pids)} subject files in './hr_1hz'. Starting batch processing...\n")
    
#     subject_results = {}
    
#     # 2. Process every subject through Step 2 -> Step 3 -> Step 4
#     for pid in all_pids:
#         try:
#             df_step2 = align_and_preprocess_subject(pid, use_power=True)
#             windows = cut_and_zscore_windows(df_step2, window_len=120)
#             df_xcorr = compute_subject_xcorr(windows, max_lag=60, min_windows=3)
            
#             subject_results[pid] = df_xcorr
            
#             if df_xcorr is not None:
#                 print(f"  [PASS] Subject {pid}: {df_xcorr['n_windows'].iloc[0]} windows analyzed.")
#             else:
#                 print(f"  [DROP] Subject {pid}: Insufficient windows (< 3).")
                
#         except Exception as e:
#             print(f"  [ERROR] Subject {pid} failed: {e}")
#             subject_results[pid] = None

#     print("\n-------------------------------------------")
    
#     # 3. Run Stage 2: Grand Average across all valid subjects
#     try:
#         df_panel_f = compute_grand_average(subject_results)
        
#         # Find where the grand-mean correlation peaks
#         peak_row = df_panel_f.loc[df_panel_f['grand_mean'].idxmax()]
#         print(f"\nGroup-Level Peak: r = {peak_row['grand_mean']:.4f} at lag = {int(peak_row['lag'])} s")
        
#         # 4. Plot Panel F: Grand Average Correlogram with SEM Shading
#         lags = df_panel_f['lag']
#         g_mean = df_panel_f['grand_mean']
#         g_sem = df_panel_f['sem']
        
#         plt.figure(figsize=(9, 5))
        
#         # Plot the mean curve
#         plt.plot(lags, g_mean, color='#1f77b4', lw=2.0, label='Grand Mean (N2 Sleep)')
        
#         # Plot the shaded SEM error band (Panel F style)
#         plt.fill_between(
#             lags, 
#             g_mean - g_sem, 
#             g_mean + g_sem, 
#             color='#1f77b4', 
#             alpha=0.25, 
#             label=f'± 1 SEM (N = {df_panel_f["n_subjects"].iloc[0]})'
#         )
        
#         plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Lag 0')
#         plt.axhline(0, color='black', linestyle='-', alpha=0.3)
        
#         plt.title("Panel F — Grand-Mean Heart Rate / Sigma-Band Power Cross-Correlation")
#         plt.xlabel("Lag τ (s) [Positive = Sigma lags HR]")
#         plt.ylabel("Correlation (r)")
#         plt.grid(True, alpha=0.3)
#         plt.legend(loc='upper right')
#         plt.tight_layout()
        
#         # Save the plot so you can attach it to your email to Ronny!
#         plt.savefig("Panel_F_Grand_Average_Reproduction.png", dpi=300)
#         print("Saved plot as 'Panel_F_Grand_Average_Reproduction.png'.")
#         plt.show()
        
#     except Exception as e:
#         print(f"[FATAL ERROR] Could not compute grand average: {e}")

# if __name__ == "__main__":    
#     test_pid = "200006"
    
#     try:
#         # Step 2: Align
#         df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
#         # Step 3: Window & Z-score
#         windows = cut_and_zscore_windows(df_step2, window_len=120)
#         # Step 4: Cross-Correlate & Average
#         df_xcorr = compute_subject_xcorr(windows, max_lag=60, min_windows=3)
        
#         if df_xcorr is not None:
#             print(f"--- Step 4 Within-Subject XCorr Successful for Patient {test_pid} ---")
#             print(f"Analyzed {df_xcorr['n_windows'].iloc[0]} windows across lags -60 to +60 s.")
            
#             # Find where correlation peaks
#             peak_row = df_xcorr.loc[df_xcorr['r_mean'].idxmax()]
#             print(f"\nPeak positive correlation: r = {peak_row['r_mean']:.4f} at lag = {int(peak_row['lag'])} s")
            
#             # Quick preview plot of Panel D
#             plt.figure(figsize=(8, 4))
#             plt.plot(df_xcorr['lag'], df_xcorr['r_mean'], color='black', lw=1.5)
#             plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Lag 0')
#             plt.title(f"Panel D Preview — Single-Subject Correlogram ({test_pid})")
#             plt.xlabel("Lag τ (s) [Positive = Sigma lags HR]")
#             plt.ylabel("Correlation (r)")
#             plt.grid(True, alpha=0.3)
#             plt.legend()
#             plt.tight_layout()
#             plt.show()
            
#     except Exception as e:
#         print(f"[ERROR] Could not process {test_pid}: {e}")

# # --- TEST SNIPPET ---
# if __name__ == "__main__":
#     test_pid = "200001"
    
#     try:
#         df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
#         print(f"--- Step 2 Alignment Successful for Patient {test_pid} ---")
        
#         # Run Step 3
#         windows = cut_and_zscore_windows(df_step2, window_len=120)
#         print(f"\n--- Step 3 Windowing & Z-Scoring Successful ---")
#         print(f"Total valid 120-second windows extracted: {len(windows)}")
        
#         if windows:
#             print("\nSample of Window #0 (first 5 seconds):")
#             print(windows[0].head())
#             print("\nVerification of Z-score properties for Window #0:")
#             print(f"  hr_z mean     : {windows[0]['hr_z'].mean():.6f} (should be ~0.0)")
#             print(f"  hr_z std      : {windows[0]['hr_z'].std():.6f} (should be ~1.0)")
#             print(f"  sigma_z mean  : {windows[0]['sigma_z'].mean():.6f} (should be ~0.0)")
#             print(f"  sigma_z std   : {windows[0]['sigma_z'].std():.6f} (should be ~1.0)")
            
#     except Exception as e:
#         print(f"[ERROR] Could not process {test_pid}: {e}")

# if __name__ == "__main__":
#     test_pid = "200001"
    
#     try:
#         # Toggle use_power=True here to match Lecci et al. (2017) power units
#         df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
#         print(f"--- Step 2 Alignment Successful for Patient {test_pid} (use_power=True) ---")
#         print(df_step2.head(10))
#         print(f"\nTotal aligned N2 seconds ready for Step 3: {len(df_step2)}")
#     except Exception as e:
#         print(f"[ERROR] Could not process {test_pid}: {e}")

    