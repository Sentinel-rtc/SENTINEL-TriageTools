import numpy as np
from scipy.signal import butter, filtfilt
import argparse
import csv
import os
import scipy

class GAMBITCalculator:
    def __init__(self, frequency: float):
        """Initialize with sampling frequency (time step in seconds)."""
        self.frequency = frequency
        self.dt = frequency  # Time step (e.g., 0.001 s for 1 kHz)
        self.a_c = 250.0  # Critical translational acceleration threshold (G), per Newman (1985), page 10
        self.alpha_c = 10000.0  # Critical rotational acceleration threshold (rad/s²), per Newman (1985), page 10

    def butter_lowpass(self, cutoff: float, order: int = 4):
        """
        Create Butterworth low-pass filter coefficients.

        Args:
            cutoff (float): Cutoff frequency in Hz (400-500 Hz per Newman, page 4).
            order (int): Filter order (4th order per Newman, page 4).

        Returns:
            Tuple: Filter coefficients (b, a).
        """
        nyquist = 0.5 / self.dt  # Nyquist frequency
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a 400 Hz Butterworth filter per Newman (1985), page 4.

        Args:
            data (np.ndarray): Input data array (e.g., acceleration time series).

        Returns:
            np.ndarray: Filtered data.
        """
        b, a = self.butter_lowpass(cutoff=400.0)  # 400-500 Hz range, using 400 Hz as conservative choice
        filtered = filtfilt(b, a, data)
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
            expected_header = ['Time', 'ax', 'ay', 'az', 'alphax', 'alphay', 'alphaz']
            if len(header) != 7 or header != expected_header:
                raise ValueError("CSV must have columns: Time, ax, ay, az, alphax, alphay, alphaz")
            data = np.array([list(map(float, row)) for row in reader])
        return data

    def calculate_gambit(self, data: np.ndarray) -> float:
        """
        Calculate GAMBIT per simplified formula G = (a_m / 250) + (alpha_m / 10000) from Newman (1985), page 10.

        Args:
            data (np.ndarray): Array with columns [Time, ax, ay, az, alphax, alphay, alphaz].
                              ax, ay, az in G; alphax, alphay, alphaz in rad/s².

        Returns:
            float: GAMBIT value (G).
        """
        print("GAMBIT model used in the provided Python code is the linear method")
        # Extract acceleration components
        ax, ay, az = data[:, 1], data[:, 2], data[:, 3]
        alphax, alphay, alphaz = data[:, 4], data[:, 5], data[:, 6]

        # Apply 400 Hz filter to all components (per Newman, page 4)
        ax_filt = self.apply_filter(ax)
        ay_filt = self.apply_filter(ay)
        az_filt = self.apply_filter(az)
        alphax_filt = self.apply_filter(alphax)
        alphay_filt = self.apply_filter(alphay)
        alphaz_filt = self.apply_filter(alphaz)

        # Compute resultant accelerations at each time step
        a_resultant = np.sqrt(ax_filt**2 + ay_filt**2 + az_filt**2)  # Resultant translational acceleration (G)
        alpha_resultant = np.sqrt(alphax_filt**2 + alphay_filt**2 + alphaz_filt**2)  # Resultant rotational acceleration (rad/s²)

        # Extract maximum values (a_m and alpha_m)
        a_m = np.max(a_resultant)
        alpha_m = np.max(alpha_resultant)

        # Calculate GAMBIT (simplified linear form, page 10)
        gambit = (a_m / self.a_c) + (alpha_m / self.alpha_c)
        return gambit

    def process_file(self, file_path: str) -> float:
        """
        Process CSV file and calculate GAMBIT.

        Args:
            file_path (str): Path to CSV file.

        Returns:
            float: GAMBIT value (G).
        """
        data = self.read_csv(file_path)
        gambit_value = self.calculate_gambit(data)
        return gambit_value

if __name__ == "__main__":
    # Command-line argument parsing with default file_path
    parser = argparse.ArgumentParser(description="Calculate GAMBIT from CSV data per Newman (1985).")
    parser.add_argument("--frequency", type=float, required=True, help="Sampling frequency (time step in seconds, e.g., 0.001 for 1 kHz)")
    parser.add_argument("--file_path", type=str, 
                        default='impact_data.csv',  # Adjust default path as needed
                        help="Path to CSV file with acceleration data")
    args = parser.parse_args()

    # Initialize calculator and compute GAMBIT
    gambit_calculator = GAMBITCalculator(args.frequency)
    try:
        gambit_value = gambit_calculator.process_file(args.file_path)
        print(f"GAMBIT Value: {gambit_value:.3f}")
        if gambit_value <= 1.0:
            print("Result: No unacceptable injury (G ≤ 1)")
        else:
            print("Result: Exceeds threshold, injury likely (G > 1)")
    except Exception as e:
        print(f"Error: {e}")
        
        
    # Example usage: python GAMBIT_Calculator.py --frequency 0.001 --file_path "impact_data.csv"