from matplotlib import image, pyplot as plt
import numpy as np
from fft import FFT2D

# Load image as pixel array
img = image.imread('kingfisher.jpg')

# Create an instance of FFT2D class
fft_object = FFT2D(img[:, :, 0])  # Assuming you want to sharpen the red channel
fft_object2=FFT2D(img[:,:,1])
fft_object3=FFT2D(img[:,:,2])

# Apply the sharpening process on the red color channel separately
fft2d_red_channel = fft_object.fft2d()
fft2d_green_channel = fft_object2.fft2d()
fft2d_blue_channel = fft_object3.fft2d()

fft2d_red_channel = fft2d_red_channel.ifft2d()
fft2d_green_channel = fft2d_green_channel.ifft2d()
fft2d_blue_channel = fft2d_blue_channel.ifft2d()

# Plotting the superimposed channel figures
plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
plt.imshow(np.log(1 + np.abs(fft2d_red_channel)))  # Display the red channel FFT
# plt.title('Red Channel')

# plt.subplot(1, 3, 2)
plt.imshow(np.log(1 + np.abs(fft2d_green_channel)))  # Display the green channel FFT
# plt.title('Green Channel')

# plt.subplot(1, 3, 3)
plt.imshow(np.log(1 + np.abs(fft2d_blue_channel)))  # Display the blue channel FFT
plt.title('Blue Channel')

plt.tight_layout()
plt.show()