import os
import glob
import numpy as np
import pandas as pd

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
    2. Loads N2 sigma envelope data[cite: 8].
    3. Performs an inner join on integer 'time' (restricts HR exclusively to N2 sleep)[cite: 8].
    4. Normalizes sigma envelope to % of its nightly N2 mean[cite: 8].
    5. Applies a 4-second centered symmetric moving average to sigma_pct[cite: 8].
    6. Leaves instantaneous heart rate unsmoothed[cite: 8].

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
        normalizing[cite: 8]. If False, uses the raw amplitude envelope[cite: 8].
    """
    hr_path = os.path.join(hr_dir, f"{pid}.csv")
    sigma_path = os.path.join(sigma_dir, f"{pid}.csv")

    if not os.path.exists(hr_path) or not os.path.exists(sigma_path):
        raise FileNotFoundError(f"Missing input files for subject {pid}.")

    # 1. Load both traces
    df_hr = pd.read_csv(hr_path)
    df_sigma = pd.read_csv(sigma_path, usecols=['time', 'envelope'])

    # Square the envelope before normalizing if strict power units are requested[cite: 8]
    # OPTIONAL
    if use_power:
        df_sigma['envelope'] = df_sigma['envelope'] ** 2

    # 2. Calculate the nightly N2 mean BEFORE any dropping/joining[cite: 8]
    mean_n2_envelope = df_sigma['envelope'].mean()
    if mean_n2_envelope == 0 or np.isnan(mean_n2_envelope):
        raise ValueError(f"Sigma envelope mean is 0 or NaN for subject {pid}.")

    # 3. Inner join on integer 'time' -> restricts data exclusively to shared N2 sleep seconds[cite: 8]
    df_aligned = pd.merge(df_hr, df_sigma, on='time', how='inner')
    df_aligned = df_aligned.sort_values('time').reset_index(drop=True)

    # 4. Normalize sigma to % of nightly N2 mean[cite: 8]
    df_aligned['sigma_pct'] = (100.0 * df_aligned['envelope']) / mean_n2_envelope

    # 5. Apply a 4-second centered (symmetric) moving average to sigma_pct[cite: 8]
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
    # 1. Identify continuous stretches where time increments by exactly 1 second[cite: 8]
    # A jump in time (> 1 second) increments the stretch_id[cite: 8]
    # neatly use cumsum of boolean mask to create unique stretch IDs[cite: 8]
    stretch_ids = (df_clean['time'].diff() != 1).cumsum()
    
    windows = []
    
    # 2. Process each continuous stretch independently[cite: 8]
    for _, stretch in df_clean.groupby(stretch_ids):
        n_seconds = len(stretch)
        
        # If the stretch is shorter than window_len (e.g. 120 s), drop it[cite: 8]
        if n_seconds < window_len:
            continue
            
        # Calculate how many complete 120-s windows fit in this stretch[cite: 8]
        n_windows = n_seconds // window_len
        
        # Truncate any trailing remainder that doesn't fit into a full 120-s window[cite: 8]
        stretch_truncated = stretch.iloc[: n_windows * window_len].copy()
        
        # Assign a local window number (0, 1, 2, ...) within this stretch
        # neatly use the // operator to assign window IDs: 0, 1, 2, ... , window_len will be 0, then window_len + 1, window_len + 2, ... 2*window_len will get 1, etc.
        stretch_truncated['window_id'] = np.arange(len(stretch_truncated)) // window_len
        
        # 3. Z-score 'hr' and 'sigma_smooth' independently within each 120-s window[cite: 8]
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


# --- TEST SNIPPET ---
if __name__ == "__main__":
    test_pid = "200001"
    
    try:
        df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
        print(f"--- Step 2 Alignment Successful for Patient {test_pid} ---")
        
        # Run Step 3
        windows = cut_and_zscore_windows(df_step2, window_len=120)
        print(f"\n--- Step 3 Windowing & Z-Scoring Successful ---")
        print(f"Total valid 120-second windows extracted: {len(windows)}")
        
        if windows:
            print("\nSample of Window #0 (first 5 seconds):")
            print(windows[0].head())
            print("\nVerification of Z-score properties for Window #0:")
            print(f"  hr_z mean     : {windows[0]['hr_z'].mean():.6f} (should be ~0.0)")
            print(f"  hr_z std      : {windows[0]['hr_z'].std():.6f} (should be ~1.0)")
            print(f"  sigma_z mean  : {windows[0]['sigma_z'].mean():.6f} (should be ~0.0)")
            print(f"  sigma_z std   : {windows[0]['sigma_z'].std():.6f} (should be ~1.0)")
            
    except Exception as e:
        print(f"[ERROR] Could not process {test_pid}: {e}")

# if __name__ == "__main__":
#     test_pid = "200001"
    
#     try:
#         # Toggle use_power=True here to match Lecci et al. (2017) power units[cite: 8]
#         df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
#         print(f"--- Step 2 Alignment Successful for Patient {test_pid} (use_power=True) ---")
#         print(df_step2.head(10))
#         print(f"\nTotal aligned N2 seconds ready for Step 3: {len(df_step2)}")
#     except Exception as e:
#         print(f"[ERROR] Could not process {test_pid}: {e}")

    