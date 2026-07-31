import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Instead of loading all the data at once and then running some analysis,
# I will load the data one by one, calculating what I need per patient, then discard the unnecessary data.
# This will save memory and allow me to run the analysis on a larger dataset.

def process_patient_ecg(patient_id, ecg_dir='./ecg'):
    """
    Loads and processes the ECG beat file for a single patient.
    """
    # 1. Load single patient data:
    # The guideline specifies to only use the {ID}_1.ecg files and ignore _2.ecg files
    ecg_filepath = f"{ecg_dir}/{patient_id}_1.ecg"
    
    # Read the file using Pandas. 
    # skiprows=1 skips the metadata header line. 
    # Since it is a single column of integers after the header, we explicitly name it.
    df_ecg = pd.read_csv(ecg_filepath, skiprows=1, names=['r_peak_ms'])

    # 1. Chronological sort & deduplicate identical millisecond detections
    df_ecg = df_ecg.sort_values('r_peak_ms').drop_duplicates(subset=['r_peak_ms']).reset_index(drop=True)
    
    # 2. Convert timestamps from milliseconds to seconds
    # t_beat = ms / 1000
    df_ecg['t_beat_sec'] = df_ecg['r_peak_ms'] / 1000.0
    
    # 3. Calculate R-R intervals and instantaneous Heart Rate (BPM)
    # .diff() instantly subtracts the previous row's time from the current row's time. (I could write a for loop that calculates element i - element i-1, but this is much faster (important!) and more elegant.)
    # The very first row will become NaN (Not a Number) because there is no previous beat.
    df_ecg['rr_interval'] = df_ecg['t_beat_sec'].diff()
    
    # HR = 60 / RR
    df_ecg['hr_bpm'] = 60.0 / df_ecg['rr_interval']
    
    # Drop the first row since it contains NaNs from the .diff() calculation
    df_ecg = df_ecg.dropna().reset_index(drop=True)

    # 4. Reject Artifacts (Keep only 30 <= HR <= 220)
    df_ecg_clean = df_ecg[(df_ecg['hr_bpm'] >= 30) & (df_ecg['hr_bpm'] <= 220)].copy()

    # This is the recommendation from the guideline: "also drop isolated beats whose HR differs from the local median (e.g. a 5-beat window) by more than ~30%."
    # Calculate the rolling median over a 5-beat window (centered on the current beat)
    rolling_median = df_ecg_clean['hr_bpm'].rolling(window=5, center=True).median()
    
    # Calculate how far off the current beat is from its local median (as a percentage)
    pct_diff = np.abs(df_ecg_clean['hr_bpm'] - rolling_median) / rolling_median
    
    # Keep only the beats where the difference is 30% (0.30) or less
    df_ecg_clean = df_ecg_clean[pct_diff <= 0.30].copy()

    # Recompute R-R intervals so gaps created by dropped artifacts are measured correctly:
    df_ecg_clean['rr_interval'] = df_ecg_clean['t_beat_sec'].diff()
    
    # 5. Resample to a 1 Hz Grid
    # Find the very first and very last valid second in the recording
    start_sec = int(np.ceil(df_ecg_clean['t_beat_sec'].min()))
    end_sec = int(np.floor(df_ecg_clean['t_beat_sec'].max()))
    
    # Create the new 1 Hz timeline (e.g., 2, 3, 4, 5...)
    # Default spacing for np.arange is 1, so we don't need to specify it explicitly.
    t_grid = np.arange(start_sec, end_sec + 1)

    # Set timestamps as the index and heart rate as values
    ser = df_ecg_clean.set_index('t_beat_sec')['hr_bpm']
    
    # Union the original heartbeat index with the 1 Hz target grid
    # example:
    #   ser.index (irregular actual beats) = [1.15, 2.02, 2.88, 3.74]
    #   t_grid    (desired 1 Hz clock)     = [1, 2, 3, 4]
    #   combined_index (sorted union)      = [1.0, 1.15, 2.0, 2.02, 2.88, 3.0, 3.74, 4.0]
    combined_index = ser.index.union(t_grid)
    
    # Reindex to combine both timelines and apply Pandas linear index interpolation
    hr_interpolated = ser.reindex(combined_index).interpolate(method='index') # TODO: change this in the future if you do a PR to try to add a max_gap parameter to interpolate function in pandas
    
    # Extract only the 1 Hz grid points
    hr_1hz = hr_interpolated.loc[t_grid].to_numpy(copy=True) # this fixes an error, I used .values before

    # 6. The 5-Second Gap Rule (Vectorized mask)
    next_beat_idx = np.searchsorted(df_ecg_clean['t_beat_sec'], t_grid)
    next_beat_idx = np.clip(next_beat_idx, 0, len(df_ecg_clean) - 1)
    
    interval_sizes = df_ecg_clean['rr_interval'].values[next_beat_idx]
    hr_1hz[interval_sizes > 5.0] = np.nan
    
    # Return the clean DataFrame
    df_1hz = pd.DataFrame({'time': t_grid, 'hr': hr_1hz})
    return df_1hz.dropna().reset_index(drop=True)

    # Below is also the np.interp method, I will try to use pandas instead
    # Interpolate all points instantly and only then filter the long gaps out
    # I am not so comfortable with this, because we interpolate everything and then filter out long gaps instead of interpolating only the points we need.
    # However, to my knowledge it is actually the most efficient way to do it.
    # TODO: maybe open an issue in Pandas or neurokit2 or mnepython and maybe try to offer a PR
    # hr_1hz = np.interp(t_grid, 
    #                    df_ecg_clean['t_beat_sec'], 
    #                    df_ecg_clean['hr_bpm'], 
    #                    left=np.nan, 
    #                    right=np.nan)

    # IGNORE THIS:
    {
    # # Create an interpolation function based on our clean heartbeat data (resampling to 1 Hz)
    # # It basically gives a function that can "guess" (in a neat way) what the heart rate would be at the desired times (1.0, 2.0, 3.0, ...), based on the known heart rates at the aperiodic times (e.g. 1.2, 1.9, 2.1, 3.1, ...).
    # # "in a neat way" - since we are using linear interpolation, it will draw a straight line between the two nearest known points and use that to estimate the value at the desired point.
    # interp_func = interp1d(df_ecg_clean['t_beat_sec'], df_ecg_clean['hr_bpm'], 
    #                        kind='linear', bounds_error=False, fill_value=np.nan)
    
    # # Apply the function to the new 1 Hz grid
    # hr_1hz = interp_func(t_grid)
    
    # # 6. The 5-Second Gap Rule # unfortunately, there is no max_gap parameter in the interp1d function, and there will probably never be because this is actually deprecated (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) so I will actually use a different method instead. Commenting this code now...
    # # So apperantly it is better and more efficient to interpolate and then mask since masking introduces conditional branching and makes the cache and the way the CPU run less efficient. Either way I am adding a feature to pandas to be able to pass a "max_gap" like variable so that will be encapsulated within the pandas engine.
    # # since the function "will draw a straight line between the two nearest known points and use that to estimate the value at the desired point",\
    # # it will happily interpolate across gaps in the data, even if those gaps are huge (e.g., 10 seconds, 20 seconds, etc.).
    # # this is unwanted behavior; we need to find those gaps and erase the interpolated data that falls inside them.
    # df_1hz = pd.DataFrame({'time': t_grid, 'hr': hr_1hz})
    
    # # Find where the raw RR intervals were dangerously large (>5s, as mentioned in the guideline)
    # large_gaps = df_ecg_clean[df_ecg_clean['rr_interval'] > 5.0]
    
    # # Loop through the massive gaps and erase the fake interpolated data inside them
    # for _, row in large_gaps.iterrows():
    #     gap_start = row['t_beat_sec'] - row['rr_interval']
    #     gap_end = row['t_beat_sec']
    #     # Set the 'hr' column to NaN for any 1Hz timestamp falling inside this gap
    #     df_1hz.loc[(df_1hz['time'] > gap_start) & (df_1hz['time'] < gap_end), 'hr'] = np.nan
    # 
    # 
    # 
    # # Drop the NaNs to finalize the clean 1 Hz timeline
    # return df_1hz.dropna().reset_index(drop=True)
    }

    # 6. The 5-Second Gap Rule (100% Vectorized — No slow 'for' loops!)
    # Find which R-R interval each second in t_grid belongs to
    # next_beat_idx = np.searchsorted(df_ecg_clean['t_beat_sec'], t_grid)
    # next_beat_idx = np.clip(next_beat_idx, 0, len(df_ecg_clean) - 1)
    
    # # Grab the interval sizes for every second on the grid
    # interval_sizes = df_ecg_clean['rr_interval'].values[next_beat_idx]
    
    # # Instantly mask any second that fell inside a gap > 5.0 seconds
    # hr_1hz[interval_sizes > 5.0] = np.nan
    
    # # Return the clean DataFrame
    # df_1hz = pd.DataFrame({'time': t_grid, 'hr': hr_1hz})
    # return df_1hz.dropna().reset_index(drop=True)



# Assume process_patient_ecg is imported or defined above

def validate_step_1(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2'):
    """
    Runs the 4 mandatory STEP 1 QC checks on all matching subjects.
    """
    # Find all subjects who have a Sigma N2 file
    sigma_files = glob.glob(os.path.join(sigma_dir, "*.csv"))
    patient_ids = [os.path.splitext(os.path.basename(f))[0] for f in sigma_files]
    
    qc_results = []
    
    print(f"Starting STEP 1 validation for {len(patient_ids)} subjects...\n")
    
    for pid in patient_ids:
        ecg_path = os.path.join(ecg_dir, f"{pid}_1.ecg")
        sigma_path = os.path.join(sigma_dir, f"{pid}.csv")
        
        # Check if matching ECG file exists
        if not os.path.exists(ecg_path):
            print(f"[WARNING] Missing ECG file for subject {pid}: expected {ecg_path}")
            continue
            
        # 1. Process ECG
        try:
            # Process the ECG data for this patient
            df_hr = process_patient_ecg(pid, ecg_dir=ecg_dir)
        except Exception as e:
            print(f"[ERROR] Failed to process ECG for subject {pid}: {e}")
            continue
        
        # 2. Load Sigma N2 timestamps for Clock Alignment check
        df_sigma = pd.read_csv(sigma_path, usecols=['time'])
        
        # --- METRICS CALCULATION ---
        median_hr = df_hr['hr'].median()
        duration_hrs = df_hr['time'].max() / 3600.0
        
        # Clock overlap: what percentage of N2 seconds exist in our 1-Hz HR trace?
        hr_time_set = set(df_hr['time'])
        n2_time_set = set(df_sigma['time'])
        overlap_count = len(n2_time_set.intersection(hr_time_set))
        overlap_pct = (overlap_count / len(n2_time_set)) * 100.0 if len(n2_time_set) > 0 else 0
        
        qc_results.append({
            'patient_id': pid,
            'median_hr': median_hr,
            'duration_hrs': duration_hrs,
            'n2_overlap_pct': overlap_pct,
            'df_hr_clean': df_hr # Kept for plotting below
        })
        
    df_qc = pd.DataFrame(qc_results)
    
    # --- CHECK 1 & 2: SUMMARY STATS ---
    print("="*60)
    print("CHECK 1 & 2: POPULATION HEART RATE & DURATION SUMMARY")
    print("="*60)
    print(df_qc[['median_hr', 'duration_hrs', 'n2_overlap_pct']].describe().to_string())
    print("\n")
    
    # Outlier detection
    bad_hr = df_qc[(df_qc['median_hr'] < 55) | (df_qc['median_hr'] > 85)]
    if not bad_hr.empty:
        print(f"[ALERT] {len(bad_hr)} subjects have suspicious median HR (<55 or >85 bpm):")
        print(bad_hr[['patient_id', 'median_hr', 'duration_hrs']].to_string(index=False))
    else:
        print("[PASS] All subjects have physiological median resting HRs.")

    # --- CHECK 3: CLOCK ALIGNMENT ---
    print("\n" + "="*60)
    print("CHECK 3: CLOCK ALIGNMENT (ECG vs SIGMA N2)")
    print("="*60)
    bad_overlap = df_qc[df_qc['n2_overlap_pct'] < 90.0]
    if not bad_overlap.empty:
        print(f"[ALERT] {len(bad_overlap)} subjects have <90% N2 clock overlap with ECG:")
        print(bad_overlap[['patient_id', 'n2_overlap_pct']].to_string(index=False))
    else:
        print("[PASS] Clock alignment successful. N2 timestamps cleanly overlap with ECG trace.")

    # --- CHECK 4: VISUAL SANITY CHECK (3-5 Subjects) ---
    print("\n" + "="*60)
    print("CHECK 4: VISUAL INSPECTION (Plotting 3 sample subjects)")
    print("="*60)
    
    sample_subjects = df_qc.sample(min(3, len(df_qc)), random_state=42)
    
    fig, axes = plt.subplots(len(sample_subjects), 1, figsize=(12, 3 * len(sample_subjects)), sharey=True)
    if len(sample_subjects) == 1:
        axes = [axes]
        
    for ax, (_, row) in zip(axes, sample_subjects.iterrows()):
        df_sample = row['df_hr_clean']
        pid = row['patient_id']
        
        # Grab a 15-minute window (900 seconds) from the middle of the recording
        mid_time = df_sample['time'].median()
        window = df_sample[(df_sample['time'] >= mid_time) & (df_sample['time'] <= mid_time + 900)]
        
        ax.plot(window['time'] / 60.0, window['hr'], color='black', lw=1)
        ax.set_title(f"Patient {pid} — 15-Minute Sample Window (Median HR: {row['median_hr']:.1f} bpm)", fontsize=10)
        ax.set_ylabel("HR (bpm)")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (Minutes from Recording Start)")
    plt.tight_layout()
    plt.show()

    return df_qc

if __name__ == "__main__":
    # Run validation
    df_qc = validate_step_1(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2')

# if __name__ == "__main__":
#     # Test the function on the first patient
#     patient_id = "205804"
#     df_ecg = process_patient_ecg(patient_id)
    
#     print(f"--- ECG Data for Patient {patient_id} ---")
#     print(df_ecg.head())

#     with open("output.txt", "w", encoding="utf-8") as file:
#         file.write(df_ecg.to_string())