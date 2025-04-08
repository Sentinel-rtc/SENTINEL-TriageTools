import os
import csv
import pandas as pd
import time
import numpy as np
from typing import Dict, Tuple
import argparse
from scipy.signal import butter, filtfilt

class HICCalculator:
    """Calculates Head Injury Criterion (HIC) values from acceleration data.
    
    Implements ISO 6487 filtering standards and computes HIC values for various time windows.
    """
    # ISO 6487 CFC to cutoff frequency mapping (Hz)
    _CFC_TO_FC = {
        60: 100,    # CFC 60 → 100 Hz
        180: 300,   # CFC 180 → 300 Hz
        1000: 1650, # CFC 1000 → 1650 Hz
    }

    def __init__(self, file_path: str):
        """Initialize with path to CSV data file."""
        self.datafile = os.path.normpath(file_path)
        self.frequency = None  # Will be set during data processing

    def iso6487_filter(self,signal: np.ndarray, fs: float, cfc = 1000, zero_phase=True) -> np.ndarray:
        """Apply ISO 6487-compliant 4th-order Butterworth low-pass filter.
        
        Args:
            signal: Input acceleration signal (g)
            fs: Sampling frequency (Hz)
            cfc: Channel Frequency Class (60, 180, or 1000)
            
        Returns:
            Filtered signal
            
        Raises:
            ValueError: If cutoff frequency exceeds Nyquist frequency
        """
        fc = self._CFC_TO_FC[cfc]  # Cutoff frequency (Hz)
        nyquist = 0.5 * fs   # Nyquist frequency

        # Check if cutoff is feasible
        if fc >= nyquist:
            raise ValueError(f"Cutoff {fc} Hz must be < Nyquist {nyquist} Hz")
        
        # Design 4th-order Butterworth filter
        b, a = butter(N=4, Wn=fc/nyquist, btype='lowpass')
        
        # Apply zero-phase (forward-backward) filtering
        filtered_signal = filtfilt(b, a, signal)
        
        return filtered_signal

    def get_data(self, time_col: int, x_loc: int, y_loc: int, z_loc: int, cfc: int = 1000) -> Dict[int, Tuple[float, float]]:
        """Load and process acceleration data from CSV.
        
        Args:
            time_col: Time data column (1-based index)
            x_loc: X-acceleration column (1-based index)
            y_loc: Y-acceleration column (1-based index)
            z_loc: Z-acceleration column (1-based index)
            cfc: Filter class (60, 180, or 1000)
            
        Returns:
            Dict of {index: (time, filtered_magnitude)} where:
            - index: Sample index
            - time: Timestamp (s)
            - filtered_magnitude: Resultant acceleration (g)
        """
        # Load data (skip header and use specified columns)
        cols = [time_col-1, x_loc-1, y_loc-1, z_loc-1]
        raw_data = np.genfromtxt(self.datafile, delimiter=',', skip_header=1, usecols=cols)
        
        # Calculate time points and resultant magnitudes
        times = raw_data[:, 0]
        self.frequency = round(times[1] - times[0], 4)
        magnitudes = np.linalg.norm(raw_data[:, 1:], axis=1) / 9810
        
        # Apply ISO filter and package results
        filtered = self.iso6487_filter(magnitudes, 1/self.frequency, cfc)
        return {i: (round(t, 5), mag) for i, (t, mag) in enumerate(zip(times, filtered))}
    
    def get_hic_window_max(self, window_accels: np.ndarray, window_size: int, hic_s: float, start: int) -> float:
        """Calculate maximum moving average within a sliding window.
        
        Args:
            window_accels: Array of acceleration values around the peak
            window_size: Size of the averaging window in samples
            
        Returns:
            Maximum average acceleration found within the sliding windows
        """
        max_hic = 0
        best_window = ""

        # Iterate through all possible segments within window
        for seg_start in range(len(window_accels)):
            for seg_end in range(seg_start + window_size, len(window_accels) + 1):
                if seg_end - seg_start > window_size+1:
                    continue
                # Print the subarray from start to end
                segment = window_accels[seg_start:seg_end]
                if len(segment) == 1:
                    integral = segment[0]*self.frequency*0.5
                else:
                    integral = np.trapezoid(segment, dx=self.frequency)
                hic_value = (integral / hic_s) ** 2.5 * hic_s
                start_idx = start + seg_start
                end_idx = start + seg_end-1
                #print(f"Window: {window_start}:{window_end}, HIC Value: {hic_value}, window size: {segment_end-segment_start}")
                if hic_value > max_hic:
                    max_hic = hic_value
                    best_window = f"{start_idx}:{end_idx}"

        return best_window, round(max_hic,2)

    
    def calculate_hic(self, acceleration: Dict[int, Tuple[float, float]], hic_ms: float) -> Tuple[str, float]:
        hic_s = hic_ms / 1000
        window_size = int(hic_ms / (self.frequency * 1000))
        previous_window = max(1,int((hic_ms-1) / (self.frequency * 1000)))
        # Convert to numpy array for efficient processing
        accel_values = np.array([v[1] for v in acceleration.values()])

        # Find peak acceleration point
        max_idx = np.argmax(accel_values)
        
        # Extract window around the peak
        start_index = int(max(0, max_idx - window_size+1))
        end_index = int(min(len(accel_values), max_idx + window_size+1))
        window_accels = accel_values[start_index:end_index]

        window_time, maxHIC = self.get_hic_window_max(window_accels, window_size, hic_s, start_index)
        return window_time, maxHIC

def main():
    #start_time = time.time()
    parser = argparse.ArgumentParser(description="HIC Calculation Script")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--time", type=float, required=True, help="Column index for time data")
    parser.add_argument("--x_location", type=int, required=True, help="Column index for X direction data")
    parser.add_argument("--y_location", type=int, required=True, help="Column index for Y direction data")
    parser.add_argument("--z_location", type=int, required=True, help="Column index for Z direction data")

    args = parser.parse_args()

    hic_calculator = HICCalculator(args.file_path)
    acceleration_data = hic_calculator.get_data(args.time, args.x_location, args.y_location, args.z_location)

    hic_values = {}
    hic_ms = 0
    while hic_ms < 15.0:
        hic_ms += 0.1
        time_window,hic_value = hic_calculator.calculate_hic(acceleration_data, hic_ms)
        hic_values[time_window] = hic_value

    max_hic_value = max(hic_values.values())
    max_hic_window = max(hic_values, key=hic_values.get)

    #print(max_hic_value)
    print(f'The HIC 15 value is {max_hic_value} and was achieved between the time window of {max_hic_window}')
    #print("Process finished --- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
