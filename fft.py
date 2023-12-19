import numpy as np

class FFT2D:
    def __init__(self, array):
        self.array = array
        self.rows = len(array)
        # self.cols = len(array[0])

    def fft1d(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self.fft1d(x[::2])
        odd = self.fft1d(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N//2) / N)
        first_half = even + factor * odd
        second_half = even - factor * odd
        return np.concatenate([first_half, second_half])

    def apply_window(self, data):
        hann_window = np.hanning(data.shape[0])[:, np.newaxis] * np.hanning(data.shape[1])
        return data * hann_window
    
    def fft2d(self):
        image_data_windowed = self.apply_window(self.array)

        # Pad rows and columns to the nearest power of 2
        rows_power_of_2 = 2 ** int(np.ceil(np.log2(image_data_windowed.shape[0])))
        cols_power_of_2 = 2 ** int(np.ceil(np.log2(image_data_windowed.shape[1])))

        padded_array = np.zeros((rows_power_of_2, cols_power_of_2))
        padded_array[:image_data_windowed.shape[0], :image_data_windowed.shape[1]] = image_data_windowed

        # Apply 1D FFT along rows
        fft_rows = np.array([self.fft1d(row) for row in padded_array])

        # Apply 1D FFT along columns
        fft_cols = np.array([self.fft1d(fft_rows[:, i]) for i in range(cols_power_of_2)]).T

        return FFT2D(fft_cols)

    def ifft2d(self):
        # Apply 1D IFFT along columns
        ifft_cols = np.array([self.fft1d(row).conjugate() for row in self.array.T]).T

        # Apply 1D IFFT along rows
        ifft_rows = np.array([self.fft1d(col).conjugate() for col in ifft_cols.T]).T

        # Normalize the result to get the original image scale
        ifft_rows /= (self.rows * self.cols)

        return ifft_rows.real
    def fft_sharpen(self, channel_img):
        # Apply your custom sharpening process using FFT for each channel
        fft_img = self.fft2d(channel_img)

        # Define a sharpening filter (increase high-frequency components)
        sharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

        # Pad the filter to match the shape of fft_img
        padded_filter = np.zeros_like(fft_img)
        filter_rows, filter_cols = sharpening_filter.shape
        padded_filter[:filter_rows, :filter_cols] = sharpening_filter

        # Apply the sharpening filter in the frequency domain
        fft_sharpened = fft_img * padded_filter

        # Inverse FFT to obtain the sharpened image for this channel
        sharpened_channel = self.ifft2d(fft_sharpened)

        return sharpened_channel

if __name__=="__main__":


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
