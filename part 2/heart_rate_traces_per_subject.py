import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_patient_ecg(patient_id, ecg_dir='./ecg'):
    """
    Loads and processes the ECG beat file for a single patient.
    
    1. Reads cumulative R-peak detections in milliseconds.
    2. Converts to instantaneous HR (bpm) via R-R intervals.
    3. Filters out physiological artifacts (30 <= HR <= 220 bpm and <= 20% local median deviation).
    4. Resamples to a uniform 1 Hz grid via linear index interpolation.
    5. Applies the vectorized 5-second gap rule to mask sensor dropouts.
    """
    ecg_filepath = f"{ecg_dir}/{patient_id}_1.ecg"
    df_ecg = pd.read_csv(ecg_filepath, skiprows=1, names=['r_peak_ms'])

    # 1. Chronological sort & deduplicate identical millisecond detections
    df_ecg = df_ecg.sort_values('r_peak_ms').drop_duplicates(subset=['r_peak_ms']).reset_index(drop=True)
    
    # 2. Convert timestamps from milliseconds to seconds
    df_ecg['t_beat_sec'] = df_ecg['r_peak_ms'] / 1000.0
    
    # 3. Calculate R-R intervals (seconds) and instantaneous Heart Rate (bpm)
    df_ecg['rr_interval'] = df_ecg['t_beat_sec'].diff()
    df_ecg['hr_bpm'] = 60.0 / df_ecg['rr_interval']
    df_ecg = df_ecg.dropna().reset_index(drop=True)

    # 4. Reject Artifacts: Hard boundaries (30 <= HR <= 220 bpm)
    df_ecg_clean = df_ecg[(df_ecg['hr_bpm'] >= 30) & (df_ecg['hr_bpm'] <= 220)].copy()

    # Drop isolated beats deviating by > 20% from a local centered 11-beat median
    rolling_median = df_ecg_clean['hr_bpm'].rolling(window=11, center=True).median()
    pct_diff = np.abs(df_ecg_clean['hr_bpm'] - rolling_median) / rolling_median
    df_ecg_clean = df_ecg_clean[pct_diff <= 0.20].copy()

    # Recompute R-R intervals so gaps created by dropped artifact beats are accurately measured
    df_ecg_clean['rr_interval'] = df_ecg_clean['t_beat_sec'].diff()
    df_ecg_clean = df_ecg_clean.dropna().reset_index(drop=True)
    
    # 5. Resample to a 1 Hz Grid
    start_sec = int(np.ceil(df_ecg_clean['t_beat_sec'].min()))
    end_sec = int(np.floor(df_ecg_clean['t_beat_sec'].max()))
    t_grid = np.arange(start_sec, end_sec + 1)

    # Aggregate duplicate index timestamps by mean and union with the 1 Hz target grid
    ser = df_ecg_clean.set_index('t_beat_sec')['hr_bpm']
    ser = ser.groupby(level=0).mean().sort_index()
    combined_index = ser.index.union(t_grid)
    
    # Reindex and apply linear index interpolation
    # TODO (Upstream Optimization): Replace post-hoc gap masking with a native max_gap parameter
    # once supported in pandas.Series.interpolate()
    # See upstream tracking — Issue: https://github.com/pandas-dev/pandas/issues/66545
    #                       PR:    https://github.com/pandas-dev/pandas/pull/66548
    hr_interpolated = ser.reindex(combined_index).interpolate(method='index')
    hr_1hz = hr_interpolated.loc[t_grid].to_numpy(copy=True)

    # 6. The 5-Second Gap Rule (Vectorized mask for sensor dropouts / arousals)
    next_beat_idx = np.searchsorted(df_ecg_clean['t_beat_sec'], t_grid)
    next_beat_idx = np.clip(next_beat_idx, 0, len(df_ecg_clean) - 1)
    
    interval_sizes = df_ecg_clean['rr_interval'].values[next_beat_idx]
    hr_1hz[interval_sizes > 5.0] = np.nan
    
    df_1hz = pd.DataFrame({'time': t_grid, 'hr': hr_1hz})
    return df_1hz.dropna().reset_index(drop=True)


def run_step_1_pipeline(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2', output_dir='./hr_1hz', report_file='step_1_validation_report.txt', plot_filename='step_1_visual_inspection.png'):
    """
    Runs STEP 1 QC (Quality Control) checks across all subjects, writes verbose logs to report_file,
    and prints live line-by-line progress to the console.
    """
    os.makedirs(output_dir, exist_ok=True)

    sigma_files = glob.glob(os.path.join(sigma_dir, "*.csv"))
    patient_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in sigma_files])
    total_subjects = len(patient_ids)
    
    qc_results = []
    print(f"Processing {total_subjects} subjects... Saving 1-Hz HR traces to '{output_dir}/'.\n")

    for idx, pid in enumerate(patient_ids, start=1):
        ecg_path = os.path.join(ecg_dir, f"{pid}_1.ecg")
        sigma_path = os.path.join(sigma_dir, f"{pid}.csv")
        
        if not os.path.exists(ecg_path):
            print(f"[{idx:4d}/{total_subjects}] [SKIP] Subject {pid}: Missing ECG file ({ecg_path})")
            continue
            
        try:
            df_hr = process_patient_ecg(pid, ecg_dir=ecg_dir)

            # Save clean 1 Hz trace to disk
            out_filepath = os.path.join(output_dir, f"{pid}.csv")
            df_hr.to_csv(out_filepath, index=False)

            # Load N2 time mask for clock alignment QC
            df_sigma = pd.read_csv(sigma_path, usecols=['time'])

            # Calculate QC metrics
            median_hr = df_hr['hr'].median()
            duration_hrs = df_hr['time'].max() / 3600.0

            hr_time_set = set(df_hr['time'])
            n2_time_set = set(df_sigma['time'])
            overlap_count = len(n2_time_set.intersection(hr_time_set))
            overlap_pct = (overlap_count / len(n2_time_set)) * 100.0 if len(n2_time_set) > 0 else 0
            
            qc_results.append({
                'patient_id': pid,
                'median_hr': median_hr,
                'duration_hrs': duration_hrs,
                'n2_overlap_pct': overlap_pct,
                'df_hr_clean': df_hr
            })

            print(f"[{idx:4d}/{total_subjects}] [PASS] Subject {pid} | Median HR: {median_hr:5.1f} bpm | Overlap: {overlap_pct:5.1f}% | Dur: {duration_hrs:4.2f} h")

        except Exception as e:
            print(f"[{idx:4d}/{total_subjects}] [ERROR] Subject {pid} failed: {e}")
            
    df_qc = pd.DataFrame(qc_results)
    
    # --- WRITE COMPLETE REPORT TO FILE ---
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("STEP 1 VALIDATION COMPLETE REPORT (LECCI ET AL. 2017 REPRODUCTION)\n")
        f.write("="*80 + "\n\n")
        
        f.write("--- CHECK 1 & 2: POPULATION HEART RATE & DURATION SUMMARY ---\n")
        f.write(df_qc[['median_hr', 'duration_hrs', 'n2_overlap_pct']].describe().to_string())
        f.write("\n\n")
        
        bad_hr = df_qc[(df_qc['median_hr'] < 55) | (df_qc['median_hr'] > 85)]
        f.write(f"--- SUSPICIOUS MEDIAN HR (<55 or >85 bpm) ({len(bad_hr)} subjects) ---\n")
        if not bad_hr.empty:
            f.write(bad_hr[['patient_id', 'median_hr', 'duration_hrs']].to_string(index=False))
        else:
            f.write("None. All subjects within physiological resting baseline.\n")
        f.write("\n\n")
        
        bad_overlap = df_qc[df_qc['n2_overlap_pct'] < 90.0]
        f.write(f"--- LOW N2 CLOCK OVERLAP (<90%) ({len(bad_overlap)} subjects) ---\n")
        if not bad_overlap.empty:
            f.write(bad_overlap[['patient_id', 'n2_overlap_pct']].to_string(index=False))
        else:
            f.write("None. 100% of subjects have >=90% clock overlap.\n")
        f.write("\n\n")
        
        f.write("--- FULL COHORT METRICS TABLE ---\n")
        f.write(df_qc[['patient_id', 'median_hr', 'duration_hrs', 'n2_overlap_pct']].to_string(index=False))

    print(f"\nValidation complete! Report saved to '{report_file}'.")
    
    # --- CHECK 4: SAVE VISUAL INSPECTION PLOT ---
    sample_subjects = df_qc.sample(min(3, len(df_qc)), random_state=42)
    fig, axes = plt.subplots(len(sample_subjects), 1, figsize=(14, 3.5 * len(sample_subjects)), sharey=True)
    if len(sample_subjects) == 1:
        axes = [axes]
        
    for ax, (_, row) in zip(axes, sample_subjects.iterrows()):
        df_sample = row['df_hr_clean']
        pid = row['patient_id']
        
        mid_time = df_sample['time'].median()
        window = df_sample[(df_sample['time'] >= mid_time) & (df_sample['time'] <= mid_time + 900)]
        
        ax.plot(window['time'] / 60.0, window['hr'], color='black', lw=0.9)
        ax.set_title(f"Patient {pid} — 15-Minute Sample Window (Median HR: {row['median_hr']:.1f} bpm)", fontsize=10)
        ax.set_ylabel("HR (bpm)")
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Time (Minutes from Recording Start)")
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)
    plt.show()
    print(f"Visual inspection plot saved to '{plot_filename}'.")

    return df_qc

if __name__ == "__main__":
    df_qc = run_step_1_pipeline(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2',
                                output_dir='./hr_1hz', report_file='step_1_validation_report.txt',
                                plot_filename='step_1_visual_inspection.png')