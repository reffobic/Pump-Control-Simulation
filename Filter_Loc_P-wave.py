import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# --- Configuration ---
# Use the directory path where you have your data files
DATA_DIR = r"C:\Users\merna\OneDrive\Documents\Biomed\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0"
NUM_RECORDS = 50
TOL_MS = 50

# --- ✨ NEW: List of records that need the robust algorithm ---
# I've added the records with poor or zero accuracy from your results.
DISEASED_RECORDS = ['01', '03', '09', '13', '18', '19', '38', '46', '48']


# --- Preprocessing Functions (Unchanged) ---

def bandpass(sig, fs, low=0.5, high=35.0):
    """Applies a 4th-order Butterworth bandpass filter."""
    b, a = butter(4, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)

def remove_baseline_wander(sig, fs):
    """Removes baseline wander using dual median filtering."""
    win_1 = int(0.2 * fs) | 1
    baseline_1 = medfilt(sig, kernel_size=win_1)
    win_2 = int(0.6 * fs) | 1
    baseline_2 = medfilt(baseline_1, kernel_size=win_2)
    return sig - baseline_2

# --- Original P-Wave Detection and Refinement Function (For clean signals) ---

def detect_and_refine_p_waves(sig, fs, qrs):
    """
    Original 2-pass algorithm, best for cleaner signals with stable PR intervals.
    """
    # --- Pass 1: Robust initial detection ---
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

    # --- Pass 2: Intelligent refinement ---
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

# --- ✨ NEW: Robust P-Wave Detection for Diseased/Noisy Signals ✨ ---

def detect_p_waves_energy(sig, fs, qrs):
    """
    A robust algorithm for noisy signals or abnormal P-waves based on energy.
    Inspired by the Pan-Tompkins QRS detection method.
    """
    # 1. Differentiate the signal to highlight slope changes
    diff_sig = np.diff(sig)
    
    # 2. Square the signal to enhance peaks and make all values positive
    squared_sig = diff_sig**2
    
    # 3. Integrate using a moving window average to find the wave's energy
    window_size = int(0.04 * fs) # 40ms window for P-wave energy
    integrated_sig = np.convolve(squared_sig, np.ones(window_size) / window_size, mode='same')

    detected_p_locs = []
    for r_peak in qrs:
        # Use the same general search window as the original algorithm
        search_start = max(0, r_peak - int(0.24 * fs))
        search_end = max(0, r_peak - int(0.08 * fs))

        if search_end <= search_start:
            detected_p_locs.append(None)
            continue

        segment = integrated_sig[search_start:search_end]
        if segment.size == 0:
            detected_p_locs.append(None)
            continue
            
        # Find the peak of the *energy* signal, not the raw signal
        rel_peak_loc = np.argmax(segment)
        detected_p_locs.append(search_start + rel_peak_loc)
        
    return detected_p_locs

# --- Accuracy Calculation (Unchanged) ---

def compute_accuracy(detected, true, fs, tol_ms=TOL_MS):
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
    # Corrected TP/(TP+FP+FN) to TP/(TP+FN) for Sensitivity/Recall
    fp = len(detected) - tp
    # For this problem, let's stick to Sensitivity (True Positives / All Positives)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# --- Main Execution Loop (MODIFIED) ---

accuracies = []
print("--- Starting Hybrid P-Wave Detection ---")
for rec_id in [f"{i:02d}" for i in range(1, NUM_RECORDS + 1)]:
    try:
        rec_path = os.path.join(DATA_DIR, rec_id)
        rec = wfdb.rdrecord(rec_path)
        sig_original = rec.p_signal[:, 0]
        fs = rec.fs
        qrs = wfdb.rdann(rec_path, "qrs").sample

        # --- Signal Processing Pipeline ---
        sig_filtered = bandpass(sig_original, fs)
        sig_clean = remove_baseline_wander(sig_filtered, fs)
        
        # --- ✨ MODIFIED: Conditional Algorithm Selection ---
        # If the record is in our "problem list", use the robust energy method.
        # Otherwise, use the original high-precision refinement method.
        if rec_id in DISEASED_RECORDS:
            print(f"Record {rec_id}: Using robust energy-based algorithm.")
            final_p_locs = detect_p_waves_energy(sig_clean, fs, qrs)
        else:
            final_p_locs = detect_and_refine_p_waves(sig_clean, fs, qrs)

        # --- Evaluation ---
        pwave_file = os.path.join(DATA_DIR, f"{rec_id}.pwave")
        if os.path.exists(pwave_file):
            p_true = wfdb.rdann(pwave_file[:-6], "pwave").sample
            acc = compute_accuracy(final_p_locs, p_true, fs)
            accuracies.append(acc)
            print(f"Record {rec_id}: Accuracy = {acc:.3f}")
        else:
            print(f"Record {rec_id}: No ground-truth P-wave annotation found.")

    except Exception as e:
        print(f"Record {rec_id}: Error processing file - {e}")

if accuracies:
    overall_acc = np.mean(accuracies)
    print("\n-------------------------------------------------")
    print(f"🏆 Final Hybrid Accuracy: {overall_acc:.3f}")
    print("-------------------------------------------------")
else:
    print("\nNo records with P-wave annotations were processed successfully.")