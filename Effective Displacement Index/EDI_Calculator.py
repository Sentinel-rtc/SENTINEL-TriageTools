import os
import csv
from typing import Dict, Tuple
from scipy.signal import butter, lfilter
import argparse
import time

import pandas as pd
import numpy as np
from scipy.integrate import cumulative_trapezoid

class EDICalculator:
    """Calculates the Effective Displacement Index (EDI) for head impact assessment from acceleration data.
    
    The EDI is a metric that evaluates potential head injury risk by analyzing displacement
    resulting from impact acceleration data. It compares the maximum head displacement against
    a critical threshold (45.72mm) to determine injury risk level.
    """
    
    def __init__(self, frequency: float, cutoff: float = 1650.0, order: int = 2):
        """Initialize the EDI calculator with signal processing parameters.
        
        Args:
            frequency: Sampling frequency of the input data (in seconds)
            cutoff: Cutoff frequency for low-pass filter (in Hz, default 1650Hz)
            order: Order of Butterworth filter (default 2)
        """
        self.frequency = frequency
        self.cutoff = cutoff
        self.order = order

    def butter_lowpass(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design a Butterworth low-pass filter for signal processing.
        
        Returns:
            Tuple containing numerator (b) and denominator (a) polynomials 
            of the IIR filter
        """
        nyquist = 0.5 / self.frequency  # Nyquist Frequency
        normal_cutoff = self.cutoff / nyquist  # Normalized cutoff frequency
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def butter_lowpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply low-pass filter to raw acceleration data.
        
        Args:
            data: Raw acceleration magnitude values
            
        Returns:
            Filtered acceleration data with high-frequency noise removed
        """
        b, a = self.butter_lowpass()
        return lfilter(b, a, data)
    
    def get_file(self, file_path: str) -> str:
        """Normalize file path separators for cross-platform compatibility."""
        return file_path.replace(os.sep, os.path.sep)
    
    def get_data(self, path: str, x_loc: int, y_loc: int, z_loc: int) -> Dict[int, Tuple[float, float]]:
        """Load and preprocess triaxial acceleration data from CSV file.
        
        Processes raw acceleration data by:
        1. Calculating resultant magnitude from X/Y/Z components
        2. Applying low-pass filter to remove noise
        3. Creating time series with specified sampling frequency
        
        Args:
            path: Path to CSV file containing acceleration data
            x_loc: Column index for X-axis acceleration (1-based)
            y_loc: Column index for Y-axis acceleration (1-based)
            z_loc: Column index for Z-axis acceleration (1-based)
            
        Returns:
            Dictionary mapping sample index to (timestamp, filtered_magnitude) pairs
        """
        # Read data in one go for better performance
        with open(path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            raw_data = np.array([list(map(float, row)) for row in reader], dtype=float)
        
        # Calculate resultant magnitudes from X/Y/Z components
        times = np.arange(len(raw_data)) * self.frequency
        magnitudes = np.linalg.norm(raw_data[:, [x_loc-1, y_loc-1, z_loc-1]], axis=1)
        
        # Apply low-pass filter to remove high-frequency noise
        filtered_magnitudes = self.butter_lowpass_filter(magnitudes)
        
        # Create time-indexed dictionary of filtered magnitudes
        return {
            i: (round(t, 5), mag) 
            for i, (t, mag) in enumerate(zip(times, filtered_magnitudes))
        }
    
    def calculate_edi(self, acceleration: Dict[int, Tuple[float, float]]) -> Tuple[str, float]:
        """Compute Energy Diffusive Index (EDI) from filtered acceleration data.
        
        The calculation involves:
        1. Numerical integration of acceleration to get velocity
        2. Numerical integration of velocity to get displacement
        3. Finding maximum displacement during impact
        4. Calculating EDI as ratio to critical displacement (45.72mm)
        
        Args:
            acceleration: Time-indexed filtered acceleration data
            
        Returns:
            Tuple containing:
            - Maximum head displacement (mm)
            - Computed EDI value
            - Impact classification ("Hazardous" or "associated with concussion")
        """
        # Convert to numpy arrays for vectorized processing
        accel_values = np.array([v[1] for v in acceleration.values()])
        
        # Perform numerical integration to derive displacement
        velocity = cumulative_trapezoid(accel_values, dx=self.frequency, initial=0)
        displacement = cumulative_trapezoid(velocity, dx=self.frequency, initial=0)
        
        # Find peak displacement during impact event
        max_displacement = round(np.max(displacement), 2)

        # Calculate EDI as ratio to critical threshold (45.72mm)
        EDI = round(max_displacement / 45.72, 5)
        impact = "Hazardous" if EDI > 1 else "associated with concussion"
        return max_displacement, EDI, impact
    
def main():
    """Command-line interface for EDI calculation tool."""
    parser = argparse.ArgumentParser(
        description="Effective Displacement Index (EDI) Calculator for Head Impact Assessment"
    )
    parser.add_argument("--frequency", type=float, required=True, 
                       help="Sampling frequency of data (in seconds)")
    parser.add_argument("--file_path", type=str, required=True, 
                       help="Path to CSV file containing acceleration data")
    parser.add_argument("--x_location", type=int, required=True, 
                       help="1-based column index for X-axis acceleration data")
    parser.add_argument("--y_location", type=int, required=True, 
                       help="1-based column index for Y-axis acceleration data")
    parser.add_argument("--z_location", type=int, required=True, 
                       help="1-based column index for Z-axis acceleration data")

    args = parser.parse_args()
    
    # Process data and compute EDI metrics
    calculator = EDICalculator(args.frequency)
    accel_data = calculator.get_data(
        args.file_path.replace(os.sep, os.path.sep),
        args.x_location,
        args.y_location,
        args.z_location
    )
    headDisplacement, EDI, impact = calculator.calculate_edi(accel_data)

    print(f"With head displacement of {headDisplacement}mm and EDI of {EDI}, the impact is {impact}")

if __name__ == "__main__":
    main()