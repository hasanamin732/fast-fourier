import numpy as np

class FFT2D:
    def __init__(self, array):
        self.array = array
        self.rows = array.shape[0]
        self.cols = array.shape[1]
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

    def fft2d(self,shape=None):
        # Pad rows and columns to the nearest power of 2
        rows_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[0])))
        cols_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[1])))

        padded_array = np.zeros((rows_power_of_2, cols_power_of_2))
        padded_array[:self.array.shape[0], :self.array.shape[1]] = self.array

        # Apply 1D FFT along rows
        fft_rows = np.array([self.fft1d(row) for row in padded_array])
        # fft_rows = np.array([self.fft1d(row) for row in self.array])

        # Apply 1D FFT along columns
        fft_cols = np.array([self.fft1d(fft_rows[:, i]) for i in range(cols_power_of_2)]).T


        if shape is not None:
            if shape[0] <= self.rows and shape[1] <= self.cols:
                fft_cols = fft_cols[:shape[0], :shape[1]]
            elif shape[0] > self.rows and shape[1] <= self.cols:
                # Shape exceeds rows, but fits within columns
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[:self.rows, :self.cols] = fft_cols[:self.rows, :shape[1]]
                fft_cols = zeros
            if shape[0] <= self.rows and shape[1] > self.cols:
                # Shape fits within rows, but exceeds columns
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[:self.rows, :self.cols] = fft_cols[:shape[0], :self.cols]
                fft_cols = zeros
            else:
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[:self.rows, :self.cols] = fft_cols[:self.rows, :self.cols]
                fft_cols = zeros



        # self.array = fft_cols
        return FFT2D(fft_cols)


    def ifft1d(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.ifft1d(x[::2])
        odd = self.ifft1d(x[1::2])
        factor = np.exp(2j * np.pi * np.arange(N//2) / N)  # Reverse the sign for inverse
        first_half = even + factor * odd
        second_half = even - factor * odd
        return np.concatenate([first_half, second_half]) / 2  # Adjust the scaling factor

    def ifft2d(self,original_shape):
        # Apply 1D inverse FFT along columns
        ifft_cols = np.array([self.ifft1d(col) for col in self.array.T])

        # Apply 1D inverse FFT along rows
        ifft_rows = np.array([self.ifft1d(row) for row in ifft_cols.T])
        cropped_ifft = ifft_rows[:original_shape[0], :original_shape[1]]

        return np.real(cropped_ifft)
        # return np.real(ifft_rows)
    # def ifft2d(self, original_shape):
    #     rows_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[0])))
    #     cols_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[1])))

    #     # Pad rows and columns to the nearest power of 2
    #     padded_array = np.zeros((rows_power_of_2, cols_power_of_2), dtype=np.complex128)
    #     padded_array[:self.array.shape[0], :self.array.shape[1]] = self.array

    #     # Apply 1D IFFT along columns
    #     ifft_cols = np.array([self.ifft1d(col) for col in padded_array.T]).T

    #     # Apply 1D IFFT along rows
    #     ifft_rows = np.array([self.ifft1d(row) for row in ifft_cols.T]).T

    #     # Crop the result to the original shape
    #     cropped_ifft = ifft_rows[:original_shape[0], :original_shape[1]]

    #     return np.real(cropped_ifft)


    def __str__(self):
        return str(self.array)
