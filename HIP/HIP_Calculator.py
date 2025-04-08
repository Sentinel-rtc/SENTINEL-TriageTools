import numpy as np
from scipy.signal import butter, filtfilt
import argparse
import csv
import os
import scipy

class HIPCalculator:
    def __init__(self, frequency: float):
        """Initialize with sampling frequency (time step in seconds)."""
        self.frequency = frequency
        self.dt = frequency  # Time step (e.g., 0.0001 s for 10 kHz)

    def butter_lowpass(self, cutoff: float, order: int = 2):
        """
        Create Butterworth low-pass filter coefficients.

        Args:
            cutoff (float): Cutoff frequency in Hz (e.g., 1650 for CFC 1000, 300 for CFC 180).
            order (int): Filter order (default 2).

        Returns:
            Tuple: Filter coefficients (b, a).
        """
        nyquist = 0.5 / self.dt  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """
        Apply CFC 1000 (1650 Hz) and CFC 180 (300 Hz) filters sequentially.

        Args:
            data (np.ndarray): Input data array (e.g., acceleration time series).

        Returns:
            np.ndarray: Filtered data.
        """
        # First filter: CFC 1000 (1650 Hz)
        b1, a1 = self.butter_lowpass(cutoff=1650.0)
        filtered = filtfilt(b1, a1, data)
        # Second filter: CFC 180 (300 Hz)
        b2, a2 = self.butter_lowpass(cutoff=300.0)
        filtered = filtfilt(b2, a2, filtered)
        return filtered

    @staticmethod
    def read_csv(file_path: str) -> np.ndarray:
        """
        Read CSV file with Time, ax, ay, az, alphax, alphay, alphaz columns.

        Args:
            file_path (str): Path to CSV file.

        Returns:
            np.ndarray: Data array with shape (n_samples, 7).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            if len(header) != 7 or header != ['Time', 'ax', 'ay', 'az', 'alphax', 'alphay', 'alphaz']:
                raise ValueError("CSV must have columns: Time, ax, ay, az, alphax, alphay, alphaz")
            data = np.array([list(map(float, row)) for row in reader])
        return data

    def manual_cumtrapz(self, y: np.ndarray, x: np.ndarray = None) -> np.ndarray:
        """
        Manually implement cumulative trapezoidal integration.
        
        Args:
            y (np.ndarray): Data to integrate (e.g., ax, ay, az, etc.)
            x (np.ndarray): Time values or independent variable for integration (default: np.arange(len(y))).

        Returns:
            np.ndarray: Cumulative integration of y.
        """
        if x is None:
            x = np.arange(len(y))  # Default: Use the index as the x values
        dx = np.diff(x)  # Compute the differences between consecutive x values
        integral = np.insert(np.cumsum((y[:-1] + y[1:]) * dx / 2), 0, 0)  # Cumulative sum with trapezoidal rule
        return integral

    def calculate_hip(self, data: np.ndarray) -> float:
        """
        Calculate Head Impact Power (HIP) per Equation 7 (PAGE 4).

        Args:
            data (np.ndarray): Array with columns [Time, ax, ay, az, alphax, alphay, alphaz].

        Returns:
            float: Maximum HIP value (HIP_m) in kW.
        """
        # Extract acceleration components
        time = data[:, 0]
        ax, ay, az = data[:, 1], data[:, 2], data[:, 3]
        alphax, alphay, alphaz = data[:, 4], data[:, 5], data[:, 6]

        # Apply filters to all components (CFC 1000 + CFC 180)
        ax_filt = self.apply_filters(ax)
        ay_filt = self.apply_filters(ay)
        az_filt = self.apply_filters(az)
        alphax_filt = self.apply_filters(alphax)
        alphay_filt = self.apply_filters(alphay)
        alphaz_filt = self.apply_filters(alphaz)

        # Compute integrals (velocity changes) using manual cumulative trapezoidal rule
        int_ax = self.manual_cumtrapz(ax_filt, time)
        int_ay = self.manual_cumtrapz(ay_filt, time)
        int_az = self.manual_cumtrapz(az_filt, time)
        int_alphax = self.manual_cumtrapz(alphax_filt, time)
        int_alphay = self.manual_cumtrapz(alphay_filt, time)
        int_alphaz = self.manual_cumtrapz(alphaz_filt, time)

        # HIP formula coefficients (PAGE 4)
        m = 4.50  # Mass of head (kg)
        Ix, Iy, Iz = 0.016, 0.024, 0.022  # Moments of inertia (Nms²)

        # Calculate HIP at each time step
        hip = (m * ax_filt * int_ax + 
               m * ay_filt * int_ay + 
               m * az_filt * int_az + 
               Ix * alphax_filt * int_alphax + 
               Iy * alphay_filt * int_alphay + 
               Iz * alphaz_filt * int_alphaz) / 1000  # Convert W to kW

        # Return maximum HIP (HIP_m)
        return np.max(hip)

    def process_file(self, file_path: str) -> float:
        """
        Process CSV file and calculate HIP.

        Args:
            file_path (str): Path to CSV file.

        Returns:
            float: Maximum HIP value (HIP_m) in kW.
        """
        data = self.read_csv(file_path)
        hip_m = self.calculate_hip(data)
        return hip_m

if __name__ == "__main__":
    # Command-line argument parsing with default file_path
    parser = argparse.ArgumentParser(description="Calculate GAMBIT from CSV data per Newman (1985).")
    parser.add_argument("--frequency", type=float, required=True, help="Sampling frequency (time step in seconds, e.g., 0.001 for 1 kHz)")
    parser.add_argument("--file_path", type=str, 
                        default='impact_data.csv',  # Adjust default path as needed
                        help="Path to CSV file with acceleration data")
    args = parser.parse_args()

    # Initialize calculator and compute HIP
    hip_calculator = HIPCalculator(args.frequency)
    try:
        hip_value = hip_calculator.process_file(args.file_path)
        print(f"Maximum Head Impact Power (HIP_m): {hip_value:.2f} kW")
    except Exception as e:
        print(f"Error: {e}")
        
        
  #python HIP_Calculator.py --frequency 0.0001 --file_path "impact_data.csv"