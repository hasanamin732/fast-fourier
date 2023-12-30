import numpy as np
from fft import FFT2D

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    x, y = np.meshgrid(np.linspace(-1, 1, size[0]), np.linspace(-1, 1, size[1]))
    d = np.sqrt(x*x + y*y)
    kernel = np.exp(-(d**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_low_pass_filter(image, sigma,kernel_size=None):
    """Apply Gaussian low-pass filter to the image."""
    if kernel_size is None:
        kernel_size=(int(6*sigma+1),int(6*sigma+1))
    kernel = gaussian_kernel(kernel_size, sigma)
    image_obj=FFT2D(image)
    kernel_obj=FFT2D(kernel)
    # Perform Fourier Transform of the image and the kernel
    image_fft = image_obj.fft2d()
    kernel_fft = kernel_obj.fft2d(n=image.shape)
    
    # Multiply in the frequency domain
    filtered_image_fft = image_fft * kernel_fft
    filtered_image_fft_Obj=FFT2D(filtered_image_fft)
    # Inverse Fourier Transform to get the filtered image
    filtered_image = np.abs(filtered_image_fft_Obj.ifft2d(image.shape))
    
    return filtered_image

