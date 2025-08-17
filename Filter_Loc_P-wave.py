import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# Check for required libraries
try:
    import wfdb
    import pywt
except ImportError:
    print("A required library is not installed.")
    print("Please install 'wfdb' and 'PyWavelets' by running:")
    print("pip install wfdb pywavelets")
    exit()

# Configuration for BUT-PDB database
CONFIG = {
    'record_name': '03',  # Record number (01-50)
    'db_name': 'but-pdb',
    'description': 'Brno University of Technology P-Wave DB'
}

BARE_MINIMUM_BPM = 60  # Fallback heart rate if no P-wave is detected

def load_real_ecg_data(config):
    """Load ECG data and expert annotations from PhysioNet."""
    record_name = config['record_name']
    db = config['db_name']
    print(f"Step 1: Loading data for record '{record_name}' from config '{db}'...")
    
    # Load signal and annotations
    record = wfdb.rdrecord(record_name, pn_dir='but-pdb/1.0.0/')
    r_peak_ann = wfdb.rdann(record_name, 'qrs', pn_dir='but-pdb/1.0.0/')
    p_wave_ann = wfdb.rdann(record_name, 'pwave', pn_dir='but-pdb/1.0.0/')
    
    signal = record.p_signal[:, 0]  # Use first ECG lead
    fs = record.fs
    expert_r_peaks = r_peak_ann.sample
    expert_p_waves = p_wave_ann.sample
        
    print("Data loaded successfully.")
    return signal, expert_r_peaks, expert_p_waves, fs

def enhance_pwave_wavelet(sig, fs):
    """Enhance P-wave using wavelet transform to remove noise."""
    print("Step 3: Filtering and enhancing signal using Wavelet Transform...")
    coeffs = pywt.wavedec(sig, 'db4', level=5)
    
    # Zero out unwanted frequency components
    coeffs_keep = [c.copy() for c in coeffs]
    coeffs_keep[0][:] = 0   # Remove low frequency
    coeffs_keep[-1][:] = 0  # Remove highest frequency
    coeffs_keep[-2][:] = 0  # Remove second highest frequency

    enhanced = pywt.waverec(coeffs_keep, 'db4')

    # Fix length mismatch from transform
    if len(enhanced) > len(sig):
        enhanced = enhanced[:len(sig)]
    elif len(enhanced) < len(sig):
        enhanced = np.pad(enhanced, (0, len(sig) - len(enhanced)), mode="edge")
        
    return enhanced

def detect_p_waves_adaptive_algorithm(enhanced_signal, r_peaks, fs):
    """Detect P-waves using adaptive window based on RR-interval."""
    print("Step 4: Running adaptive P-wave detection algorithm...")
    detected_p_waves = []

    for i, r_peak in enumerate(r_peaks):
        # Estimate RR interval for adaptive window
        if i > 0:
            rr_interval_sec = (r_peak - r_peaks[i-1]) / fs
        else:
            rr_interval_sec = np.median(np.diff(r_peaks)) / fs

        # Define adaptive PR search window
        pr_max_sec = max(0.12, 0.20 * (rr_interval_sec / 1.0))
        pr_min_sec = max(0.08, 0.12 * (rr_interval_sec / 1.0))

        # Convert to sample indices
        start_search = int(r_peak - pr_max_sec * fs)
        end_search = int(r_peak - pr_min_sec * fs)

        if start_search >= 0 and end_search > start_search:
            search_window = enhanced_signal[start_search:end_search]
            if len(search_window) > 0:
                # Find P-wave as maximum in search window
                p_location_relative = np.argmax(search_window)
                detected_p_waves.append(start_search + p_location_relative)
    
    print(f"Algorithm detected {len(detected_p_waves)} P-waves.")
    return np.array(detected_p_waves)

def generate_pump_drive_signal(num_samples, p_waves, r_peaks, fs):
    """Generate pump drive signal based on detected P-waves."""
    print("Step 6 & 7: Generating pump drive signal...")
    pump_signal = np.zeros(num_samples)
    last_r_peak = 0
    bare_minimum_interval_samples = int((60.0 / BARE_MINIMUM_BPM) * fs)
    
    for r_peak in r_peaks:
        p_wave_in_cycle = np.any((p_waves > last_r_peak) & (p_waves < r_peak))
        if p_wave_in_cycle:
            p_wave_index = p_waves[(p_waves > last_r_peak) & (p_waves < r_peak)][0]
            drive_time_index = p_wave_index + int(0.02 * fs)
            if drive_time_index < len(pump_signal):
                pump_signal[drive_time_index:drive_time_index+10] = 1
        else:
            drive_time_index = last_r_peak + bare_minimum_interval_samples
            if drive_time_index < r_peak and drive_time_index < len(pump_signal):
                 pump_signal[drive_time_index:drive_time_index+10] = 0.5
        last_r_peak = r_peak
    return pump_signal

def plot_results(time, original, enhanced, r_peaks, detected_p, expert_p, pump_signal, config):
    """Plot results for comparison and debugging."""
    print("Step 8: Plotting results for debugging and comparison...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f"ECG Simulation: {config['description']} (Record: {config['record_name']})", fontsize=16)

    # Original vs enhanced signal
    axes[0].set_title('Original vs. Wavelet-Enhanced Signal')
    axes[0].plot(time, original, label='Original Real ECG Signal', color='lightgray', alpha=0.9)
    axes[0].plot(time, enhanced, label='Wavelet-Enhanced Signal', color='blue', linewidth=1.5)
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylabel('Amplitude (mV)')

    # P-wave detection comparison
    axes[1].set_title('P-Wave Detection vs. Expert Annotation')
    axes[1].plot(time, enhanced, label='Enhanced Signal', color='blue')
    axes[1].plot(time[r_peaks], enhanced[r_peaks], "x", label='Expert R-Peaks', color='red', markersize=8)
    axes[1].plot(time[detected_p], enhanced[detected_p], "o", label='Algorithm-Detected P-Waves', color='lime', markersize=10, alpha=0.8)
    axes[1].plot(time[expert_p], enhanced[expert_p], "v", label='Expert-Annotated P-Waves', color='black', markersize=6)
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_ylabel('Amplitude (mV)')

    # Pump drive signal
    axes[2].set_title('Final Pump Drive Signal (Based on Detected P-Waves)')
    axes[2].plot(time, pump_signal, label='Pump Drive Signal (1=P-wave, 0.5=Fallback)', color='purple')
    axes[2].fill_between(time, 0, pump_signal, color='purple', alpha=0.3)
    axes[2].legend()
    axes[2].grid(True)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Pump Command')
    axes[2].set_ylim(-0.1, 1.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    # Load ECG data
    original_ecg, expert_r_peaks, expert_p_waves, SAMPLING_RATE = load_real_ecg_data(CONFIG)
    
    # Enhance signal using wavelet transform
    enhanced_ecg = enhance_pwave_wavelet(original_ecg, SAMPLING_RATE)

    # Detect P-waves using adaptive algorithm
    detected_p_waves = detect_p_waves_adaptive_algorithm(enhanced_ecg, expert_r_peaks, SAMPLING_RATE)

    # Generate pump drive signal
    num_samples = len(original_ecg)
    pump_drive_signal = generate_pump_drive_signal(num_samples, detected_p_waves, expert_r_peaks, fs=SAMPLING_RATE)
    
    # Plot results for 10-second segment
    time_vector = np.arange(num_samples) / SAMPLING_RATE
    end_sample = min(int(10 * SAMPLING_RATE), num_samples)
    
    plot_results(time_vector[:end_sample], 
                 original_ecg[:end_sample], 
                 enhanced_ecg[:end_sample], 
                 expert_r_peaks[expert_r_peaks < end_sample], 
                 detected_p_waves[detected_p_waves < end_sample],
                 expert_p_waves[expert_p_waves < end_sample],
                 pump_drive_signal[:end_sample],
                 CONFIG)
