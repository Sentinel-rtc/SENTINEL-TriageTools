import numpy as np
from scipy import interpolate

import os
import csv
from typing import Dict, Tuple
from scipy.signal import butter, lfilter
import argparse
import time

class WSTCCalculator:
    def __init__(self, frequency: float, cutoff: float = 1650.0, order: int = 2):
        self.frequency = frequency
        self.cutoff = cutoff
        self.order = order

        # Precompute WSTC curve interpolation
        self.wstc_x = np.array([1.25, 2.6, 4.27, 7.39, 12.9, 13.94, 28.41, 32.15, 42.46, 56.71, 71.18, 85.54, 99.9, 107.18])
        self.wstc_y = np.array([590.69, 500.48, 400.24, 300.72, 198.33, 187.59, 108.83, 100.24, 85.92, 72.32, 64.44, 60.14, 39.38, 39.38])
        self.wstc_spline = interpolate.make_interp_spline(self.wstc_x, self.wstc_y)

    def get_wstc_value(self, x_new: float) -> float:
        """Get WSTC value using precomputed spline interpolation."""
        y_new = self.wstc_y[-1] if x_new >= 107 else float(self.wstc_spline(x_new))
        return y_new

    def butter_lowpass(self):
        """
        Creates a Butterworth low-pass filter.

        Returns:
            Tuple: Filter coefficients (b, a).
        """
        nyquist = 0.5 / self.frequency  # Nyquist Frequency
        normal_cutoff = self.cutoff / nyquist  # Normalized cutoff frequency
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data):
        """
        Applies a Butterworth low-pass filter to the data.

        Args:
            data (list): The data to filter.

        Returns:
            list: Filtered data.
        """
        b, a = self.butter_lowpass()
        return lfilter(b, a, data)

    def get_file(self, file_path: str) -> str:
        """
        Returns the path to the CSV file.
        """
        return file_path.replace(os.sep, os.path.sep)

    def get_data(self, path: str, x_loc: int, y_loc: int, z_loc: int) -> Dict[int, Tuple[float, float]]:
        """
        Read and process acceleration data from CSV file.
        
        Returns:
            Dict with time step as key and (time, filtered_magnitude) as value
        """
        # Read data in one go for better performance
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            raw_data = np.array([list(map(float, row)) for row in reader], dtype=float)
        
        # Calculate magnitudes
        times = np.arange(len(raw_data)) * self.frequency
        magnitudes = np.linalg.norm(raw_data[:, [x_loc-1, y_loc-1, z_loc-1]], axis=1) / 9810
        
        # Filter magnitudes
        filtered_magnitudes = self.butter_lowpass_filter(magnitudes)
        
        # Create dictionary
        return {
            i: (round(t, 5), mag) 
            for i, (t, mag) in enumerate(zip(times, filtered_magnitudes))
        }

    def get_window_max(self, window_accels: np.ndarray, window_size: int) -> float:
        """
        Optimized sliding window maximum average calculation using numpy.
        """ 
        # Use convolution for efficient sliding window average
        averages = []
        for i in range(window_size*2):
            window = window_accels[i:i + window_size]
            avg = sum(window) / window_size
            averages.append(avg)
        max_index = averages.index(max(averages))
        return averages[max_index]


    def calculate_wstc(self, acceleration: Dict[int, Tuple[float, float]], wstc_ms: float) -> Tuple[str, float]:
        """
        Calculate WSTC impact assessment for given window size.
        
        Returns:
            "Dangerous" or "Safe" based on comparison with WSTC curve
        """
        # Convert to numpy arrays for faster processing
        accel_values = np.array([v[1] for v in acceleration.values()])
        
        # Find maximum acceleration point
        max_idx = np.argmax(accel_values)
        window_size = int(wstc_ms / (self.frequency * 1000))
        
        # Calculate window bounds
        start = max(0, max_idx - window_size)
        end = min(len(accel_values), max_idx + window_size)
        window_accels = accel_values[start:end]
        
        # Get maximum window average and compare with WSTC
        avg_accel = self.get_window_max(window_accels, window_size)
        impact = "Dangerous" if self.get_wstc_value(wstc_ms) < avg_accel else "Safe"
        return avg_accel, impact


def main():
    #start_time = time.time()
    
    parser = argparse.ArgumentParser(description="HIC Calculation Script")
    parser.add_argument("--frequency", type=float, required=True, help="Sampling frequency (e.g., 0.00001)")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--x_location", type=int, required=True, help="Column index for X direction data")
    parser.add_argument("--y_location", type=int, required=True, help="Column index for Y direction data")
    parser.add_argument("--z_location", type=int, required=True, help="Column index for Z direction data")

    args = parser.parse_args()
    
    # Initialize calculator and process data
    calculator = WSTCCalculator(args.frequency)
    accel_data = calculator.get_data(
        args.file_path.replace(os.sep, os.path.sep),
        args.x_location,
        args.y_location,
        args.z_location
    )
    
    # Find critical WSTC window size
    critical_ms = 1
    prev_avgAccel = 0
    while critical_ms < 200:
        avgAccel, impact = calculator.calculate_wstc(accel_data, critical_ms)
        if impact == "Dangerous":
            break
        if prev_avgAccel == round(avgAccel,3):
            break
        else:
            prev_avgAccel = round(avgAccel,3)
        critical_ms += 1
    
    print(f"As per WSTC this impact is: {impact}")
    #print(f"Execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()