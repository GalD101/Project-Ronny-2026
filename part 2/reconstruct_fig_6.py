import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def process_patient_ecg(patient_id, ecg_dir='./ecg'):
    """
    Loads and processes the ECG beat file for a single patient
    with upgraded artifact rejection (11-beat median + step-change filter).
    """
    ecg_filepath = f"{ecg_dir}/{patient_id}_1.ecg"
    df_ecg = pd.read_csv(ecg_filepath, skiprows=1, names=['r_peak_ms'])
    
    # 1. Chronological sort & deduplicate identical millisecond detections
    df_ecg = df_ecg.sort_values('r_peak_ms').drop_duplicates(subset=['r_peak_ms']).reset_index(drop=True)
    
    # 2. Convert to seconds & calculate instantaneous HR
    df_ecg['t_beat_sec'] = df_ecg['r_peak_ms'] / 1000.0
    df_ecg['rr_interval'] = df_ecg['t_beat_sec'].diff()
    df_ecg['hr_bpm'] = 60.0 / df_ecg['rr_interval']
    df_ecg = df_ecg.dropna().reset_index(drop=True)

    # 3. UPGRADED ARTIFACT REJECTION
    # A. Hard physiological boundaries [30, 220 bpm]
    df_ecg_clean = df_ecg[(df_ecg['hr_bpm'] >= 30) & (df_ecg['hr_bpm'] <= 220)].copy()

    # B. Wider local median filter (11-beat window, 20% deviation threshold)
    rolling_median = df_ecg_clean['hr_bpm'].rolling(window=11, center=True).median()
    pct_diff = np.abs(df_ecg_clean['hr_bpm'] - rolling_median) / rolling_median
    df_ecg_clean = df_ecg_clean[pct_diff <= 0.20].copy()

    # C. Step-change filter (drop single-beat jumps > 20 bpm)
    step_change = np.abs(df_ecg_clean['hr_bpm'].diff())
    df_ecg_clean = df_ecg_clean[step_change <= 20.0].copy()

    # Recompute R-R intervals across removed artifacts
    df_ecg_clean['rr_interval'] = df_ecg_clean['t_beat_sec'].diff()
    df_ecg_clean = df_ecg_clean.dropna().reset_index(drop=True)
    
    # 4. Resample to 1 Hz Target Grid
    start_sec = int(np.ceil(df_ecg_clean['t_beat_sec'].min()))
    end_sec = int(np.floor(df_ecg_clean['t_beat_sec'].max()))
    t_grid = np.arange(start_sec, end_sec + 1)

    ser = df_ecg_clean.set_index('t_beat_sec')['hr_bpm']
    ser = ser.groupby(level=0).mean().sort_index()
    
    combined_index = ser.index.union(t_grid)
    hr_interpolated = ser.reindex(combined_index).interpolate(method='index')
    hr_1hz = hr_interpolated.loc[t_grid].to_numpy(copy=True)

    # 5. The 5-Second Gap Rule
    next_beat_idx = np.searchsorted(df_ecg_clean['t_beat_sec'], t_grid)
    next_beat_idx = np.clip(next_beat_idx, 0, len(df_ecg_clean) - 1)
    
    interval_sizes = df_ecg_clean['rr_interval'].values[next_beat_idx]
    hr_1hz[interval_sizes > 5.0] = np.nan
    
    df_1hz = pd.DataFrame({'time': t_grid, 'hr': hr_1hz})
    return df_1hz.dropna().reset_index(drop=True)


def validate_step_1(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2', report_file='step_1_validation_report.txt'):
    """
    Runs STEP 1 QC checks on all subjects, writes verbose logs to report_file,
    and prints concise progress to the console.
    """
    sigma_files = glob.glob(os.path.join(sigma_dir, "*.csv"))
    patient_ids = sorted([os.path.splitext(os.path.basename(f))[0] for f in sigma_files])
    
    qc_results = []
    print(f"Processing {len(patient_ids)} subjects... Detailed report will be saved to '{report_file}'.")

    for pid in patient_ids:
        ecg_path = os.path.join(ecg_dir, f"{pid}_1.ecg")
        sigma_path = os.path.join(sigma_dir, f"{pid}.csv")
        
        if not os.path.exists(ecg_path):
            continue
            
        try:
            df_hr = process_patient_ecg(pid, ecg_dir=ecg_dir)
            df_sigma = pd.read_csv(sigma_path, usecols=['time'])
            
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
        except Exception as e:
            print(f"[ERROR] Skipping patient {pid} due to processing error: {e}")
            
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

    print(f"Validation complete! Report saved to '{report_file}'.")
    
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
    plot_filename = 'step_1_visual_inspection.png'
    plt.savefig(plot_filename, dpi=150)
    plt.show()
    print(f"Visual inspection plot saved to '{plot_filename}'.")

    return df_qc

if __name__ == "__main__":
    df_qc = validate_step_1(ecg_dir='./ecg', sigma_dir='./Sigma_Envelope_N2')