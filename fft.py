import numpy as np
import numpy.typing as npt


class FFT2D:
    def __init__(self, array:npt.NDArray):
        self.array = array
        self.rows = array.shape[0]
        self.cols = array.shape[1]
        

    def _fft1d(self, x):
        '''
        Applies 1-dimensional Fast Fourier Transform (FFT) using recursive Cooley-Tukey algorithm.

        Parameters
        ----------
        - x (numpy.ndarray): Input 1D numpy array.

        Returns:
        - numpy.ndarray: FFT of the input array.

        This method recursively computes the 1D FFT of the input array using the Cooley-Tukey algorithm. It splits the input array into even and odd indices, computes their respective FFTs, and combines them to produce the final FFT result.
        '''
        N = len(x)
        if N <= 1:
            return x
        even = self._fft1d(x[::2])
        odd = self._fft1d(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N // 2) / N)
        first_half = even + factor * odd
        second_half = even - factor * odd
        return np.concatenate([first_half, second_half])

    def fft2d(self, shape=None):
        '''
        Applies 2-dimensional Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.

        Parameters
        ----------
        - shape (tuple, optional): Target shape of the FFT result. If provided, the result will be adjusted to fit this shape.

        Returns:
        - FFT2D: An object with the 2D FFT array.

        If shape is specified and:
        - Both dimensions are smaller or equal to the original array, the resulting FFT will be cropped.
        - One dimension exceeds the original array, the result will be zero-padded along that dimension.
        - Both dimensions exceed the original array, the result will be zero-padded to fit the specified shape.
        '''
        # Pad rows and columns to the nearest power of 2
        rows_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[0])))
        cols_power_of_2 = 2 ** int(np.ceil(np.log2(self.array.shape[1])))

        padded_array = np.zeros((rows_power_of_2, cols_power_of_2))
        padded_array[: self.array.shape[0], : self.array.shape[1]] = self.array

        # Apply 1D FFT along rows
        fft_rows = np.array([self._fft1d(row) for row in padded_array])
        # fft_rows = np.array([self.fft1d(row) for row in self.array])

        # Apply 1D FFT along columns
        fft_cols = np.array(
            [self._fft1d(fft_rows[:, i]) for i in range(cols_power_of_2)]
        ).T
        fft_cols=FFT2D(fft_cols).fftshift()
        if shape is not None:
            if shape[0] <= self.rows and shape[1] <= self.cols:
                fft_cols = fft_cols[: shape[0], : shape[1]]
            elif shape[0] > self.rows and shape[1] <= self.cols:
                # Shape exceeds rows, but fits within columns
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[: self.rows, : self.cols] = fft_cols[: self.rows, : shape[1]]
                fft_cols = zeros
            if shape[0] <= self.rows and shape[1] > self.cols:
                # Shape fits within rows, but exceeds columns
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[: self.rows, : self.cols] = fft_cols[: shape[0], : self.cols]
                fft_cols = zeros
            else:
                zeros = np.zeros(shape, dtype=np.complex128)
                zeros[: self.rows, : self.cols] = fft_cols[: self.rows, : self.cols]
                fft_cols = zeros
        # self.array = fft_cols
        return FFT2D(fft_cols)

    def ifft1d(self, x):
        '''
        Applies 1-dimensional Inverse Fast Fourier Transform (IFFT) using a recursive algorithm.

        Parameters
        ----------
        - x (numpy.ndarray): Input 1D numpy array.

        Returns:
        - numpy.ndarray: IFFT of the input array.

        This method recursively computes the 1D IFFT of the input array using a recursive algorithm. It splits the input array into even and odd indices, computes their respective IFFTs, and combines them to produce the final IFFT result. The signs of the factor are reversed to perform the inverse operation.
        '''
        N = len(x)
        if N <= 1:
            return x
        even = self.ifft1d(x[::2])
        odd = self.ifft1d(x[1::2])
        # Reverse the sign for inverse
        factor = np.exp(2j * np.pi * np.arange(N // 2) / N)
        first_half = even + factor * odd
        second_half = even - factor * odd
        return np.concatenate([first_half, second_half]) / 2

    def ifft2d(self, original_shape):
        '''
        Applies 2-dimensional Inverse Fast Fourier Transform (IFFT) to a 2D array.

        Parameters
        ----------
        - original_shape (tuple): The original shape of the input array before padding.

        Returns:
        - numpy.ndarray: 2D array with the original shape after applying IFFT.
        '''
        # Apply 1D inverse FFT along columns
        ifft_cols = np.array([self.ifft1d(col) for col in self.array.T])

        # Apply 1D inverse FFT along rows
        ifft_rows = np.array([self.ifft1d(row) for row in ifft_cols.T])
        # ifft_rows=FFT2D(ifft_rows).ifftshift()
        cropped_ifft = ifft_rows[: original_shape[0], : original_shape[1]]

        return np.real(cropped_ifft)
        # return np.real(ifft_rows)

    def fftshift(self):
        '''
        Performs a shift of the FFT array to center the zero frequency component.

        Returns:
        - numpy.ndarray: Shifted FFT array.
        '''
        mid_row,mid_col=self.rows//2,self.cols//2
        shifted_rows = np.concatenate((self.array[mid_row:, :], self.array[:mid_row, :]), axis=0)
        shifted_arr = np.concatenate((shifted_rows[:, mid_col:], shifted_rows[:, :mid_col]), axis=1)
    
        # return FFT2D(shifted_arr)
        return shifted_arr
    
    def ifftshift(self):
        '''
        Undoes the shift performed by fftshift to restore the original FFT array.

        Returns:
        - numpy.ndarray: Unshifted FFT array.
        '''
        mid_row,mid_col=self.rows//2,self.cols//2
         # Undo the shift along columns
        shifted_cols = np.concatenate((self.array[:, -mid_col:], self.array[:, :-mid_col]), axis=1)
        
        # Undo the shift along rows
        unshifted_arr = np.concatenate((shifted_cols[-mid_row:, :], shifted_cols[:-mid_row, :]), axis=0)
    
        return FFT2D(unshifted_arr)
    
    def __str__(self):
        '''
        Returns a string representation of the FFT2D object's array.

        Returns:
        - str: String representation of the array.
        '''
        return str(self.array)
    def __mul__(self,obj):
        '''
        Multiplies two FFT2D objects element-wise.

        Parameters
        ----------
        - obj (FFT2D): Another FFT2D object to perform element-wise multiplication.

        Returns:
        - FFT2D: Result of element-wise multiplication.
        '''
        return FFT2D(self.array*obj.array)