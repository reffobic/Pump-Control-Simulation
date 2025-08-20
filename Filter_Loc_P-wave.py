import os
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# --- Configuration ---
# ***** UPDATE THIS PATH *****
DATA_DIR = r"C:\Users\merna\OneDrive\Documents\Biomed\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0"
NUM_RECORDS = 50
TOL_MS = 50  # Tolerance for accuracy calculation in milliseconds

# --- Preprocessing Functions ---

def bandpass(sig, fs, low=0.5, high=35.0):
    """Applies a 4th-order Butterworth bandpass filter. High-end is lowered to reduce noise."""
    b, a = butter(4, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)

def remove_baseline_wander(sig, fs):
    """Removes baseline wander using dual median filtering."""
    win_1 = int(0.2 * fs) | 1
    baseline_1 = medfilt(sig, kernel_size=win_1)
    win_2 = int(0.6 * fs) | 1
    baseline_2 = medfilt(baseline_1, kernel_size=win_2)
    return sig - baseline_2

# --- P-Wave Detection ---

def detect_p_robust(sig, fs, qrs):
    """
    P-wave detection using a direct search in a defined window.
    This function operates on a pre-filtered signal.
    """
    p_locs = []
    
    for r_peak in qrs:
        # Define a wider, more reliable search window based on physiological PR interval.
        # Window: from 240ms to 80ms before the R-peak.
        search_win_start = max(0, r_peak - int(0.24 * fs))
        search_win_end = max(0, r_peak - int(0.08 * fs))

        if search_win_end <= search_win_start:
            p_locs.append(None)
            continue
        
        # Extract the segment of the signal where the P-wave is expected
        search_segment = sig[search_win_start:search_win_end]
        
        if search_segment.size == 0:
            p_locs.append(None)
            continue

        # Find the index of the absolute maximum value in the segment.
        # This works for both positive and inverted P-waves.
        # We use np.argmax on the absolute value to find the peak's location,
        # whether it's upright or inverted.
        p_peak_relative_loc = np.argmax(np.abs(search_segment))
        
        # Convert the relative location back to the signal's absolute index
        p_locs.append(search_win_start + p_peak_relative_loc)
        
    return p_locs

# --- Accuracy Calculation ---

def compute_accuracy(detected, true, fs, tol_ms=TOL_MS):
    """Computes detection accuracy (Sensitivity or TP / (TP + FN))."""
    tol_samples = int(tol_ms * fs / 1000)
    detected = sorted([d for d in detected if d is not None])
    true = sorted(true)

    if not true:
        return 1.0 if not detected else 0.0

    tp = 0  # True Positives
    used_true_indices = set()

    # Match each detected peak to the closest true peak within the tolerance
    for d_peak in detected:
        best_match_idx = -1
        min_dist = float('inf')
        
        for i, t_peak in enumerate(true):
            dist = abs(d_peak - t_peak)
            if dist <= tol_samples and dist < min_dist:
                min_dist = dist
                best_match_idx = i
        
        if best_match_idx != -1 and best_match_idx not in used_true_indices:
            tp += 1
            used_true_indices.add(best_match_idx)
    
    fn = len(true) - tp # False Negatives are true peaks that were not detected
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return sensitivity

# --- Main Execution Loop ---

accuracies = []
print("--- Starting P-Wave Detection Analysis (Robust Version) ---")
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
        
        # Use the new, robust detection function
        p_locs = detect_p_robust(sig_clean, fs, qrs)

        # --- Evaluation ---
        pwave_file = os.path.join(DATA_DIR, f"{rec_id}.pwave")
        if os.path.exists(pwave_file):
            p_true = wfdb.rdann(pwave_file[:-6], "pwave").sample
            acc = compute_accuracy(p_locs, p_true, fs, tol_ms=TOL_MS)
            accuracies.append(acc)
            print(f"Record {rec_id}: Accuracy = {acc:.3f}")
        else:
            print(f"Record {rec_id}: No ground-truth P-wave annotation found.")

    except Exception as e:
        print(f"Record {rec_id}: Error processing file - {e}")

if accuracies:
    overall_acc = np.mean(accuracies)
    print("\n-------------------------------------------------")
    print(f"✅ Overall P-Wave Detection Accuracy: {overall_acc:.3f}")
    print("-------------------------------------------------")
else:
    print("\nNo records with P-wave annotations were processed successfully.")

# --- Visualization for Debugging ---
print("\nGenerating visualization for Record '01'...")
rec_id = "01"
rec_path = os.path.join(DATA_DIR, rec_id)
rec = wfdb.rdrecord(rec_path)
sig = rec.p_signal[:, 0]
fs = rec.fs
qrs = wfdb.rdann(rec_path, "qrs").sample

# Rerun the processing pipeline for the plot
sig_filtered = bandpass(sig, fs)
sig_clean = remove_baseline_wander(sig_filtered, fs)
p_locs_debug = detect_p_robust(sig_clean, fs, qrs)

t = np.arange(len(sig)) / fs

# Plotting
plt.figure(figsize=(18, 6))
plt.plot(t, sig_clean, label="Cleaned ECG Signal", color='blue', alpha=0.8)
plt.plot(t[p_locs_debug], sig_clean[p_locs_debug], 'ro', markersize=8, label="Detected P-Peaks")
plt.plot(t[qrs], sig_clean[qrs], 'x', color='purple', markersize=10, label="QRS Peaks")
plt.title("P-Wave Detection on Cleaned Signal (Record 01)", fontsize=14)
plt.xlabel("Time (s)", fontsize=12)
plt.ylabel("Amplitude (mV)", fontsize=12)
plt.legend()
plt.grid(linestyle='--', alpha=0.6)
plt.xlim(10, 15) # Zoom into a 5-second window for clarity
plt.show()