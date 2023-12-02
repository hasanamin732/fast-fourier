import numpy as np

class FFT2D:
    def __init__(self, array):
        self.array = array
        self.rows = len(array)
        self.cols = len(array[0])

    def fft1d(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.fft1d(x[0::2])
        odd = self.fft1d(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        T = factor[:N // 2] * odd
        return np.concatenate([even + T, even - T])

    def fft2d(self):
        # Apply 1D FFT along rows
        fft_rows = np.array([self.fft1d(row) for row in self.array])

        # Apply 1D FFT along columns
        fft_cols = np.array([self.fft1d(fft_rows[:, i]) for i in range(self.cols)]).T

        return fft_cols

    def ifft2d(self, array):
        # Apply 1D IFFT along columns
        ifft_cols = np.array([self.fft1d(array[:, i].conjugate() / self.rows).conjugate() for i in range(self.cols)]).T

        # Apply 1D IFFT along rows
        ifft_rows = np.array([self.fft1d(ifft_cols[:, i].conjugate() / self.cols).conjugate() for i in range(self.rows)]).T

        return ifft_rows

# Example usage
# Create a 2D array (replace this with your data)
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

fft_object = FFT2D(data)

# Compute the 2D FFT
result_fft = fft_object.fft2d()
print("2D FFT Result:")
print(result_fft)

# Compute the 2D IFFT
result_ifft = fft_object.ifft2d(result_fft)
print("\n2D IFFT Result:")
print(result_ifft)
