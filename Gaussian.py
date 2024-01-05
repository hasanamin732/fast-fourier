from scipy.ndimage import gaussian_filter
import numpy as np
from fft import FFT2D

def gaussian_kernel(size, sigma, truncate=4.0):
    """Generate a 2D Gaussian kernel."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    d = np.sqrt(x**2 + y**2)
    kernel = np.exp(-(d**2) / (2.0 * sigma**2))
    
    # Truncate the kernel
    mask = kernel >= np.exp(-0.5 * truncate**2)
    kernel = kernel * mask
    
    return kernel / np.sum(kernel)

def gaussian_low_pass_filter(image, sigma, kernel_size=None, truncate=4.0, ifft=True):
    """Apply Gaussian low-pass filter to the image."""
    if kernel_size is None:
        kernel_size = (int(6 * np.ceil(sigma) + 1), int(6 * np.ceil(sigma) + 1))
    
    kernel = gaussian_kernel(kernel_size, sigma, truncate=truncate)
    image_obj = FFT2D(image)
    kernel_obj = FFT2D(kernel)
    
    # Perform Fourier Transform of the image and the kernel
    image_fft = image_obj.fft2d()
    kernel_fft = kernel_obj.fft2d(shape=image_fft.array.shape)
    
    # Multiply in the frequency domain by Convolution Theorem
    filtered_image_fft = image_fft * kernel_fft
    # filtered_image_fft_obj = FFT2D(filtered_image_fft)
    
    # Inverse Fourier Transform to get the filtered image
    if ifft:
        filtered_image = np.real(filtered_image_fft.ifft2d(image.shape))
        return filtered_image
    else:
        return filtered_image_fft, image_fft

def filter(image,Do=10,ifft=True):
    image_obj = FFT2D(image)
    image_fft = image_obj.fft2d()
    M,N=image_fft.array.shape
    H=np.zeros((M,N),dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D=np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v]=np.exp(-D**2/(2*Do**2))
    filtered_image_fft=(image_fft*FFT2D(H)).ifftshift()

    if ifft:
        filtered_image = np.abs(filtered_image_fft.ifft2d(image.shape))
        return filtered_image
    else:
        return filtered_image_fft, image_fft



def scipy_gaussian_low_pass_filter(image, sigma, ifft=True):
    """Apply Scipy Gaussian low-pass filter using custom FFT and IFFT."""
    image_obj = FFT2D(image)
    
    # Perform FFT of the image
    image_fft = image_obj.fft2d()
    
    # Perform Gaussian filtering in the frequency domain
    filtered_image_fft = gaussian_filter(np.real(image_fft.array), sigma)
    
    if ifft:
        # Perform IFFT to get the filtered image
        filtered_image = np.real(FFT2D(filtered_image_fft).ifft2d(image.shape))
        return filtered_image
    else:
        return filtered_image_fft

