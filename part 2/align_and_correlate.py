import os
import glob
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    test_pid = "200001"
    
    try:
        # Toggle use_power=True here to match Lecci et al. (2017) power units[cite: 8]
        df_step2 = align_and_preprocess_subject(test_pid, use_power=True)
        print(f"--- Step 2 Alignment Successful for Patient {test_pid} (use_power=True) ---")
        print(df_step2.head(10))
        print(f"\nTotal aligned N2 seconds ready for Step 3: {len(df_step2)}")
    except Exception as e:
        print(f"[ERROR] Could not process {test_pid}: {e}")

    