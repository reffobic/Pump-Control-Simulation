import os
import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
DATA_DIR = r"C:\Users\Omar\Downloads\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0"
NUM_RECORDS = 50
TOL_MS = 50  # tolerance ms

def bandpass(sig, fs, low=0.5, high=40.0):
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)

def enhance_pwave(sig, fs):
    """Wavelet-based P-wave enhancement."""
    coeffs = pywt.wavedec(sig, 'db4', level=5)
    coeffs_keep = [None] * len(coeffs)

    # keep mid-level details (avoid lowest & highest freq noise)
    for i in range(1, len(coeffs)-1):
        coeffs_keep[i] = coeffs[i]

    enhanced = pywt.waverec(coeffs_keep, 'db4')

    # fix mismatch length
    if len(enhanced) > len(sig):
        enhanced = enhanced[:len(sig)]
    elif len(enhanced) < len(sig):
        enhanced = np.pad(enhanced, (0, len(sig)-len(enhanced)), mode="edge")

    return enhanced

def detect_p(sig, fs, qrs, enhance_func):
    #Detect P-wave locations using adaptive PR window + wavelet-enhanced signal.
    enhanced = enhance_func(sig, fs)
    p_locs = []

    for i, r in enumerate(qrs):
        # Estimate RR interval
        if i > 0:
            rr = (r - qrs[i - 1]) / fs  # seconds
        else:
            rr = np.median(np.diff(qrs) / fs)  # fallback median RR

        # Adaptive PR interval window
        if rr < 0.7:      # fast HR
            pr_min, pr_max = 0.08, 0.15
        elif rr > 1.2:    # slow HR
            pr_min, pr_max = 0.12, 0.20
        else:             # normal HR
            pr_min, pr_max = 0.10, 0.18

        start = max(0, int(r - pr_max * fs))
        end   = max(0, int(r - pr_min * fs))

        if end > start:
            seg = enhanced[start:end]
            if len(seg) > 0:
                # Peak detection with prominence
                peaks, props = find_peaks(
                    seg,
                    distance=int(0.04*fs),           # min 40 ms apart
                    prominence=np.std(seg)*0.3       # avoid tiny blips
                )

                if len(peaks) > 0:
                    # choose closest to ~160ms before QRS
                    target = r - int(0.16*fs)
                    best_peak = min(peaks, key=lambda p: abs((start+p) - target))
                    p_locs.append(start + best_peak)
                else:
                    # fallback: fixed offset
                    p_locs.append(r - int(0.16*fs))
            else:
                p_locs.append(None)
        else:
            p_locs.append(None)

    return p_locs

def compute_accuracy(detected, true, fs, tol_ms=TOL_MS):
    tol_samples = int(tol_ms * fs / 1000)
    detected = sorted([d for d in detected if d is not None])
    true = sorted(true)

    matched = 0
    used = set()

    for d in detected:
        for i, t in enumerate(true):
            if i in used:
                continue
            if abs(d - t) <= tol_samples:
                matched += 1
                used.add(i)
                break

    accuracy = matched / len(true) if len(true) > 0 else 0
    return accuracy

accuracies = []

for rec_id in [f"{i:02d}" for i in range(1, NUM_RECORDS+1)]:
    try:
        rec_path = os.path.join(DATA_DIR, rec_id)
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal[:, 0]   # lead 0 only
        fs = rec.fs
        qrs = wfdb.rdann(rec_path, "qrs").sample

        # Detect P-waves
        p_locs = detect_p(sig, fs, qrs, enhance_pwave)

        # Save detections to text
        out_path = os.path.join(DATA_DIR, f"{rec_id}_p_detect_wavelet.txt")
        with open(out_path, "w") as f:
            for r, p in zip(qrs, p_locs):
                f.write(f"QRS={r}, P={p}\n")

        # Load ground-truth annotations
        pwave_file = os.path.join(DATA_DIR, f"{rec_id}.pwave")
        if os.path.exists(pwave_file):
            p_true = wfdb.rdann(pwave_file[:-6], "pwave").sample
            acc = compute_accuracy(p_locs, p_true, fs)
            accuracies.append(acc)
            print(f"{rec_id}: Accuracy={acc:.2f}")
        else:
            print(f"{rec_id}: No P-wave annotation found!")

    except Exception as e:
        print(f"{rec_id}: error {e}")

if accuracies:
    overall_acc = np.mean(accuracies)
    print(f"\nOverall P-wave detection accuracy: {overall_acc:.2f}")
else:
    print("No records with P-wave annotations found.")
#####debg
rec_id = "01"
rec = wfdb.rdrecord(os.path.join(DATA_DIR, rec_id))
sig = rec.p_signal[:, 0]
fs = rec.fs
qrs = wfdb.rdann(os.path.join(DATA_DIR, rec_id), "qrs").sample
p_locs = detect_p(sig, fs, qrs, enhance_pwave)
t = np.arange(len(sig)) / fs
enhanced = enhance_pwave(sig, fs)

plt.figure(figsize=(12, 5))
plt.plot(t, sig, label="Original", alpha=0.5)
plt.plot(t, enhanced, label="Wavelet-enhanced", alpha=0.8)
for r in qrs:
    plt.axvline(r/fs, color="r", alpha=0.2)
for p in p_locs:
    if p is not None:
        plt.plot(p/fs, enhanced[p], "go")
plt.legend()
plt.title("Record 01 P-wave detection (wavelet)")
plt.show()

