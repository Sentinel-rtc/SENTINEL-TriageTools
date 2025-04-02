import numpy as np
from scipy import interpolate
import os
import csv
from typing import Dict, Tuple
from scipy.signal import butter, lfilter
import argparse
import time

class WSTCCalculator:
    """Calculates the Wayne State Tolerance Curve (WSTC) impact assessment from acceleration data.
    
    The WSTC is a biomechanical criterion used to assess head injury risk based on 
    the relationship between impact duration and average acceleration.
    """
    
    def __init__(self, frequency: float, cutoff: float = 1650.0, order: int = 2):
        """Initialize the WSTC calculator with sampling parameters.
        
        Args:
            frequency: Sampling frequency of the acceleration data (Hz)
            cutoff: Cutoff frequency for the low-pass filter (default: 1650 Hz)
            order: Order of the Butterworth filter (default: 2)
        """
        self.frequency = frequency
        self.cutoff = cutoff
        self.order = order

        # WSTC curve data points (duration in ms vs. tolerance acceleration in g)
        self.wstc_x = np.array([1.25, 2.6, 4.27, 7.39, 12.9, 13.94, 28.41, 32.15, 
                               42.46, 56.71, 71.18, 85.54, 99.9, 107.18])
        self.wstc_y = np.array([590.69, 500.48, 400.24, 300.72, 198.33, 187.59, 
                               108.83, 100.24, 85.92, 72.32, 64.44, 60.14, 39.38, 39.38])
        
        # Create spline interpolation for the WSTC curve
        self.wstc_spline = interpolate.make_interp_spline(self.wstc_x, self.wstc_y)

    def get_wstc_value(self, x_new: float) -> float:
        """Get the tolerance acceleration value from the WSTC curve for a given duration.
        
        Args:
            x_new: Impact duration in milliseconds
            
        Returns:
            Corresponding tolerance acceleration in g from the WSTC curve
        """
        # For durations beyond 107ms, use the last known tolerance value
        y_new = self.wstc_y[-1] if x_new >= 107 else float(self.wstc_spline(x_new))
        return y_new

    def butter_lowpass(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design a Butterworth low-pass filter for signal processing.
        
        Returns:
            Tuple containing filter coefficients (b, a)
        """
        nyquist = 0.5 / self.frequency  # Nyquist Frequency
        normal_cutoff = self.cutoff / nyquist  # Normalized cutoff frequency
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply the low-pass filter to the input data.
        
        Args:
            data: Raw acceleration data to be filtered
            
        Returns:
            Filtered acceleration data
        """
        b, a = self.butter_lowpass()
        return lfilter(b, a, data)

    def get_file(self, file_path: str) -> str:
        """Normalize file path separators for cross-platform compatibility."""
        return file_path.replace(os.sep, os.path.sep)

    def get_data(self, path: str, x_loc: int, y_loc: int, z_loc: int) -> Dict[int, Tuple[float, float]]:
        """Read and process acceleration data from CSV file.
        
        Args:
            path: Path to CSV file containing acceleration data
            x_loc: Column index for X-axis acceleration
            y_loc: Column index for Y-axis acceleration
            z_loc: Column index for Z-axis acceleration
            
        Returns:
            Dictionary with time step as key and (time, filtered_magnitude) as value
            Magnitudes are converted from mm/s² to g (9.81 m/s²)
        """
        # Read and parse CSV data efficiently
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            raw_data = np.array([list(map(float, row)) for row in reader], dtype=float)
        
        # Calculate time points and resultant magnitudes
        times = np.arange(len(raw_data)) * self.frequency
        magnitudes = np.linalg.norm(raw_data[:, [x_loc-1, y_loc-1, z_loc-1]], axis=1) / 9810
        
        # Apply low-pass filter to magnitudes
        filtered_magnitudes = self.butter_lowpass_filter(magnitudes)
        
        # Package results in dictionary
        return {
            i: (round(t, 5), mag) 
            for i, (t, mag) in enumerate(zip(times, filtered_magnitudes))
        }

    def get_window_max(self, window_accels: np.ndarray, window_size: int) -> float:
        """Calculate maximum moving average within a sliding window.
        
        Args:
            window_accels: Array of acceleration values around the peak
            window_size: Size of the averaging window in samples
            
        Returns:
            Maximum average acceleration found within the sliding windows
        """
        averages = []
        for i in range(window_size*2):
            window = window_accels[i:i + window_size]
            avg = sum(window) / window_size
            averages.append(avg)
        max_index = averages.index(max(averages))
        return averages[max_index]

    def calculate_wstc(self, acceleration: Dict[int, Tuple[float, float]], wstc_ms: float) -> Tuple[float, str]:
        """Calculate WSTC impact assessment for a given window duration.
        
        Args:
            acceleration: Processed acceleration data
            wstc_ms: Window duration in milliseconds to evaluate
            
        Returns:
            Tuple containing (average_acceleration, impact_assessment)
            where impact_assessment is either "Dangerous" or "Safe"
        """
        # Convert to numpy array for efficient processing
        accel_values = np.array([v[1] for v in acceleration.values()])
        
        # Find peak acceleration point
        max_idx = np.argmax(accel_values)
        window_size = int(wstc_ms / (self.frequency * 1000))
        
        # Extract window around the peak
        start = max(0, max_idx - window_size)
        end = min(len(accel_values), max_idx + window_size)
        window_accels = accel_values[start:end]
        
        # Compare with WSTC tolerance curve
        avg_accel = self.get_window_max(window_accels, window_size)
        impact = "Dangerous" if self.get_wstc_value(wstc_ms) < avg_accel else "Safe"
        return avg_accel, impact


def main():
    """Main execution function for WSTC impact assessment."""
    
    # Configure command line arguments
    parser = argparse.ArgumentParser(description="WSTC Calculation Script")
    parser.add_argument("--frequency", type=float, required=True, help="Sampling frequency in Hz")
    parser.add_argument("--file_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--x_location", type=int, required=True, help="Column index for X acceleration")
    parser.add_argument("--y_location", type=int, required=True, help="Column index for Y acceleration")
    parser.add_argument("--z_location", type=int, required=True, help="Column index for Z acceleration")
    args = parser.parse_args()
    
    # Initialize calculator and process data
    calculator = WSTCCalculator(args.frequency)
    accel_data = calculator.get_data(
        args.file_path.replace(os.sep, os.path.sep),
        args.x_location,
        args.y_location,
        args.z_location
    )
    
    # Find critical window duration where impact becomes dangerous
    critical_ms = 1
    prev_avgAccel = 0
    while critical_ms < 200:  # Scan through durations up to 200ms
        avgAccel, impact = calculator.calculate_wstc(accel_data, critical_ms)
        if impact == "Dangerous":
            break
        if prev_avgAccel == round(avgAccel,3):  # Stop if no significant change
            break
        else:
            prev_avgAccel = round(avgAccel,3)
        critical_ms += 1
    
    print(f"As per WSTC this impact is: {impact}")

if __name__ == "__main__":
    main()