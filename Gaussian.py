import numpy as np
from fft import FFT2D


def gaussian_filter(image,sigma=10,ifft=True,low_pass=True):
    '''
    Applies a Gaussian filter to the input image in the frequency domain.

    Parameters
    ----------
    - image (numpy.ndarray): Input image array.
    - sigma (float, optional): Cut-off frequency for the Gaussian filter. Defaults to 10.
    - ifft (bool, optional): If True, performs an inverse Fourier transform (IFFT) to obtain the filtered image in spatial domain. Defaults to True.
    - low_pass (bool, optional): If True, applies a low-pass Gaussian filter. If False, applies a high-pass filter. Defaults to True.

    Returns:
    If `ifft` is True:
    - filtered_image (numpy.ndarray): Filtered image in spatial domain.

    If `ifft` is False:
    - filtered_image_fft (numpy.ndarray): Filtered image in the frequency domain.
    - image_fft (numpy.ndarray): Original image in the frequency domain.
    '''
    image_obj = FFT2D(image)
    image_fft = image_obj.fft2d()
    M,N=image_fft.array.shape
    L=np.zeros((M,N),dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D=np.sqrt((u-M/2)**2 + (v-N/2)**2)
            L[u,v]=np.exp(-D**2/(2*sigma**2))
    if low_pass:
        filtered_image_fft=(image_fft*FFT2D(L)).ifftshift()
    
    else:
        H=1-L
        filtered_image_fft=(image_fft*FFT2D(H)).ifftshift()

    if ifft:
        filtered_image = np.abs(filtered_image_fft.ifft2d(image.shape))
        return filtered_image
    else:
        return filtered_image_fft.fftshift(), image_fft



