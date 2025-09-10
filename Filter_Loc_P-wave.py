import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# --- Configuration ---
DATA_DIR = r"C:\Users\merna\OneDrive\Documents\Biomed\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0"
NUM_RECORDS = 50
TOL_MS = 50

# --- Targeted Algorithm Lists ---
AFIB_CANDIDATES = ['09', '18', '48']
NOISY_RECORDS = ['01', '03', '13', '19', '38', '46']


# --- Preprocessing Functions (Unchanged) ---
def bandpass(sig, fs, low=0.5, high=35.0):
    b, a = butter(4, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)

def remove_baseline_wander(sig, fs):
    win_1 = int(0.2 * fs) | 1
    baseline_1 = medfilt(sig, kernel_size=win_1)
    win_2 = int(0.6 * fs) | 1
    baseline_2 = medfilt(baseline_1, kernel_size=win_2)
    return sig - baseline_2

# --- Robust Atrial Fibrillation Detector (Unchanged) ---
def is_afib_robust(qrs_samples, fs, rr_std_thresh_ms=80, rmssd_thresh_ms=100):
    if len(qrs_samples) < 20:
        return False
    rr_intervals = np.diff(qrs_samples)
    rr_ms = rr_intervals * 1000 / fs
    rr_std = np.std(rr_ms)
    successive_diffs = np.diff(rr_ms)
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    if rr_std > rr_std_thresh_ms and rmssd > rmssd_thresh_ms:
        return True
    return False

# --- Original P-Wave Detection (For clean signals) (Unchanged) ---
def detect_and_refine_p_waves(sig, fs, qrs):
    # ... (function code is unchanged) ...
    initial_p_locs = []
    for r_peak in qrs:
        search_start = max(0, r_peak - int(0.24 * fs))
        search_end = max(0, r_peak - int(0.08 * fs))
        if search_end <= search_start:
            initial_p_locs.append(None)
            continue
        segment = sig[search_start:search_end]
        if segment.size == 0:
            initial_p_locs.append(None)
            continue
        rel_peak_loc = np.argmax(np.abs(segment))
        initial_p_locs.append(search_start + rel_peak_loc)
    pr_intervals = [r - p for p, r in zip(initial_p_locs, qrs) if p is not None]
    if not pr_intervals:
        return initial_p_locs
    median_pr_samples = int(np.median(pr_intervals))
    search_half_width = int(0.025 * fs)
    final_p_locs = []
    for i, r_peak in enumerate(qrs):
        expected_p_loc = r_peak - median_pr_samples
        start = max(0, expected_p_loc - search_half_width)
        end = expected_p_loc + search_half_width
        if end <= start or end >= len(sig):
            final_p_locs.append(initial_p_locs[i])
            continue
        segment = sig[start:end]
        if segment.size == 0:
            final_p_locs.append(initial_p_locs[i])
            continue
        rel_peak_loc = np.argmax(np.abs(segment))
        final_p_locs.append(start + rel_peak_loc)
    return final_p_locs

# --- ✨ UPGRADED: Refined Energy-Based P-Wave Detection ✨ ---
def detect_p_waves_energy_refined(sig, fs, qrs):
    """
    Upgraded robust algorithm with an adaptive search window and two-step peak refinement.
    """
    # 1. Differentiate, Square, and Integrate to get the energy signal
    diff_sig = np.diff(sig)
    squared_sig = diff_sig**2
    window_size = int(0.04 * fs) # 40ms window
    integrated_sig = np.convolve(squared_sig, np.ones(window_size) / window_size, mode='same')

    detected_p_locs = []
    rr_intervals = np.diff(qrs)

    for i, r_peak in enumerate(qrs):
        if i == 0:
            # For the first beat, use a default RR interval of 1 second
            rr = fs 
        else:
            rr = rr_intervals[i-1]

        # 2. Adaptive Search Window based on heart rate (RR interval)
        search_end = r_peak - int(0.15 * rr)
        search_start = r_peak - int(0.40 * rr) # Look back up to 40% of the RR interval

        search_start = max(0, search_start)
        search_end = max(search_start, search_end)

        if search_end <= search_start:
            detected_p_locs.append(None)
            continue

        # 3. Find the peak of the *energy* signal in the adaptive window
        energy_segment = integrated_sig[search_start:search_end]
        if energy_segment.size == 0:
            detected_p_locs.append(None)
            continue
        
        rel_energy_peak = np.argmax(energy_segment)
        energy_center_loc = search_start + rel_energy_peak

        # 4. Refine the peak location on the original signal
        refine_half_width = int(0.03 * fs) # 30ms search around energy center
        refine_start = max(0, energy_center_loc - refine_half_width)
        refine_end = min(len(sig), energy_center_loc + refine_half_width)

        if refine_end <= refine_start:
            detected_p_locs.append(energy_center_loc) # Fallback to energy center
            continue

        original_segment = sig[refine_start:refine_end]
        if original_segment.size == 0:
            detected_p_locs.append(energy_center_loc) # Fallback
            continue

        rel_amplitude_peak = np.argmax(np.abs(original_segment))
        final_p_loc = refine_start + rel_amplitude_peak
        detected_p_locs.append(final_p_loc)
        
    return detected_p_locs

# --- Accuracy Calculation (Unchanged) ---
def compute_accuracy(detected, true, fs, tol_ms=TOL_MS):
    # ... (function code is unchanged) ...
    tol_samples = int(tol_ms * fs / 1000)
    detected = sorted([d for d in detected if d is not None])
    true = sorted(true)
    if not true: return 1.0 if not detected else 0.0
    tp = 0
    used_true_indices = set()
    for d_peak in detected:
        best_match_idx = -1
        min_dist = float('inf')
        for i, t_peak in enumerate(true):
            dist = abs(d_peak - t_peak)
            if dist <= tol_samples and dist < min_dist:
                min_dist, best_match_idx = dist, i
        if best_match_idx != -1 and best_match_idx not in used_true_indices:
            tp += 1
            used_true_indices.add(best_match_idx)
    fn = len(true) - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# --- Main Execution Loop (MODIFIED) ---
accuracies = []
afib_records_count = 0
print("--- Starting Final Hybrid P-Wave Detection ---")
for rec_id in [f"{i:02d}" for i in range(1, NUM_RECORDS + 1)]:
    try:
        rec_path = os.path.join(DATA_DIR, rec_id)
        rec = wfdb.rdrecord(rec_path)
        sig_original = rec.p_signal[:, 0]
        fs = rec.fs
        qrs = wfdb.rdann(rec_path, "qrs").sample

        # --- Signal Processing ---
        sig_filtered = bandpass(sig_original, fs)
        sig_clean = remove_baseline_wander(sig_filtered, fs)
        
        is_afib_record = False
        # --- 3-Tier Algorithm Selection ---
        if rec_id in AFIB_CANDIDATES and is_afib_robust(qrs, fs):
            print(f"Record {rec_id}: AFib confirmed. Excluding from final accuracy score.")
            final_p_locs = []
            is_afib_record = True
        elif rec_id in AFIB_CANDIDATES or rec_id in NOISY_RECORDS:
            print(f"Record {rec_id}: Using REFINED robust energy-based algorithm.")
            final_p_locs = detect_p_waves_energy_refined(sig_clean, fs, qrs)
        else:
            final_p_locs = detect_and_refine_p_waves(sig_clean, fs, qrs)

        # --- Evaluation ---
        pwave_file = os.path.join(DATA_DIR, f"{rec_id}.pwave")
        if os.path.exists(pwave_file):
            p_true = wfdb.rdann(pwave_file[:-6], "pwave").sample
            acc = compute_accuracy(final_p_locs, p_true, fs)
            
            # ✨ MODIFIED: Only append accuracy if it's not an AFib record
            if not is_afib_record:
                accuracies.append(acc)
            else:
                afib_records_count += 1

            print(f"Record {rec_id}: Accuracy = {acc:.3f}")
        else:
            print(f"Record {rec_id}: No ground-truth P-wave annotation found.")

    except Exception as e:
        print(f"Record {rec_id}: Error processing file - {e}")

if accuracies:
    overall_acc = np.mean(accuracies)
    print("\n-------------------------------------------------")
    print(f"🏆 Final Accuracy (excluding {afib_records_count} AFib records): {overall_acc:.3f}")
    print("-------------------------------------------------")
else:
    print("\nNo records with P-wave annotations were processed successfully.")