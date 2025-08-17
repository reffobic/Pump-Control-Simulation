import os, numpy as np, wfdb, pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Config
DATA_DIR = r"C:\Users\Omar\Downloads\brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0"
GAIN = 5.0

# Filters & enhancement
def bandpass(sig, fs, low=0.5, high=40.0):
    b, a = butter(4, [low, high], btype="band", fs=fs)
    return filtfilt(b, a, sig)

def enhance_pwave(sig, fs):
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

# P-wave Detection
def detect_p(sig, fs, qrs, enhance_func):
    enhanced = enhance_func(sig, fs)
    p_locs = []

    for i, r in enumerate(qrs):
        # Estimate RR interval
        if i > 0:
            rr = (r - qrs[i - 1]) / fs  # seconds
        else:
            rr = np.median(np.diff(qrs) / fs)  # fallback to median RR

        # Adaptive PR window based on RR
        pr_min = max(0.08, 0.12 * (rr / 1.0))  # scale lower bound
        pr_max = max(0.12, 0.20 * (rr / 1.0))  # scale upper bound

        start = int(r - pr_max * fs)
        end = int(r - pr_min * fs)

        if start >= 0 and end > start:
            seg = enhanced[start:end]
            if len(seg) > 0:
                p_locs.append(start + np.argmax(seg))
            else:
                p_locs.append(None)
        else:
            p_locs.append(None)

    return p_locs


# Main loop over 50 records
for rec_id in [f"{i:02d}" for i in range(1, 51)]:
    try:
        rec = wfdb.rdrecord(os.path.join(DATA_DIR, rec_id))
        sig = rec.p_signal[:, 0]   # lead 0
        fs = rec.fs
        qrs = wfdb.rdann(os.path.join(DATA_DIR, rec_id), "qrs").sample

        p_locs = detect_p(sig, fs, qrs, enhance_pwave)
        amplified = GAIN * sig

        # save detections
        out_path = os.path.join(DATA_DIR, f"{rec_id}_p_detect_wavelet.txt")
        with open(out_path, "w") as f:
            for r, p in zip(qrs, p_locs):
                f.write(f"QRS={r}, P={p}\n")

        print(f"{rec_id}: detected {len([p for p in p_locs if p is not None])} P-waves (wavelet)")

    except Exception as e:
        print(f"{rec_id}: error {e}")

# Quick plot for record 01 (Debugging)
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
plt.legend(); plt.title("Record 01 P-wave detection (wavelet)")
plt.show()
