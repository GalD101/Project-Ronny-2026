
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


if __name__ == "__main__":
    # Test the function on the first patient
    patient_id = "205804"
    df_ecg = process_patient_ecg(patient_id)
    
    print(f"--- ECG Data for Patient {patient_id} ---")
    print(df_ecg.head())

    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(df_ecg.to_string())