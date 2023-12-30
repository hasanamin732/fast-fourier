import numpy as np

class FFT2D:
    def __init__(self, array):
        self.array = array
        self.rows = array.shape[0]
        self.cols=array.shape[1]
        self.original_shape = None

    def fft1d(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.fft1d(x[::2])
        odd = self.fft1d(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        first_half = even + factor * odd
        second_half = even - factor * odd
        return np.concatenate([first_half, second_half])

    def fft2d(self, n=None):
        # If n is not provided, use the shape of the input array
        if n is None:
            n = self.array.shape

        rows_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[0])))
        cols_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[1])))

        padded_array = np.zeros((rows_power_of_2, cols_power_of_2))
        padded_array[: self.array.shape[0], : self.array.shape[1]] = self.array

        # Apply 1D FFT along rows
        fft_rows = np.array([self.fft1d(row) for row in padded_array])

        # Apply 1D FFT along columns
        fft_cols = np.array([self.fft1d(fft_rows[:, i]) for i in range(cols_power_of_2)]).T

        if n is not None:
            result_shape = tuple(np.maximum(n[i], fft_cols.shape[i]) for i in range(2))
            cropped_fft = fft_cols[:result_shape[0], :result_shape[1]]

            padded_fft = np.zeros(n, dtype=cropped_fft.dtype)
            padded_fft[:cropped_fft.shape[0], :cropped_fft.shape[1]] = cropped_fft[
                :min(n[0], cropped_fft.shape[0]), :min(n[1], cropped_fft.shape[1])
            ]

            return padded_fft



        return fft_cols


    def ifft1d(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.ifft1d(x[::2])
        odd = self.ifft1d(x[1::2])
        # Pad arrays to have the same length
        if len(even) < len(odd):
            even = np.pad(even, (0, len(odd) - len(even)))
        elif len(odd) < len(even):
            odd = np.pad(odd, (0, len(even) - len(odd)))
        
        factor = np.exp(
            2j * np.pi * np.arange(len(even)) / len(even)
        )  # Reverse the sign for inverse
        
        # Ensure both arrays have the same length
        if len(even) != len(odd):
            raise ValueError("Even and odd arrays must have the same length.")
        
        first_half = even + factor * odd
        second_half = even - factor * odd
        return (
            np.concatenate([first_half, second_half]) / 2
        )  # Adjust the scaling factor

    def ifft2d(self, original_shape):
        # Apply 1D inverse FFT along columns
        ifft_cols = np.array([self.ifft1d(col) for col in self.array.T])

        # Apply 1D inverse FFT along rows
        ifft_rows = np.array([self.ifft1d(row) for row in ifft_cols.T])
        cropped_ifft = ifft_rows[: original_shape[0], : original_shape[1]]

        return np.real(cropped_ifft)
        # return np.real(ifft_rows)
    def __str__(self):
        return self.array



