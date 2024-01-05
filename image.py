from Gaussian import *
from matplotlib import image, pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np



image = plt.imread('rose_gray.jpg')  # Read the image

# Convert to grayscale if it's a color image
if image.ndim == 3:
    image = np.mean(image, axis=2)

# Apply the Gaussian low-pass filter
filtered_image_scipy = scipy_gaussian_low_pass_filter(image, sigma=1,ifft=True)
# filtered_image = gaussian_low_pass_filter(image, sigma=1,ifft=True)
filtered_image=filter(image,Do=10,low_pass=False)
# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Blurred Image')

plt.subplot(1, 3, 3)
plt.imshow(filtered_image_scipy, cmap='gray')
plt.title('Scipy Blurred Image')

plt.tight_layout()
plt.show()

# rows, cols = raw_fft.shape
# # freq_rows, freq_cols = np.meshgrid(np.arange(rows), np.arange(cols))
# freq_rows, freq_cols = np.meshgrid(np.fft.fftshift(np.arange(-rows//2, rows//2)), np.fft.fftshift(np.arange(-cols//2, cols//2)))
# freq_rows=freq_rows.T
# freq_cols=freq_cols.T
# fig = plt.figure(figsize=(12, 6))

# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot_surface(freq_rows, freq_cols, np.abs(raw_fft), cmap='viridis')
# ax1.set_title('Raw FFT Magnitude')
# ax1.set_xlabel('Frequency (cols)')
# ax1.set_ylabel('Frequency (rows)')
# ax1.set_zlabel('Magnitude')

# # Plotting the filtered FFT
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(freq_cols, freq_rows, np.abs(filtered_image_fft), cmap='viridis')
# ax2.set_title('Filtered FFT Magnitude')
# ax2.set_xlabel('Frequency (cols)')
# ax2.set_ylabel('Frequency (rows)')
# ax2.set_zlabel('Magnitude')

# plt.tight_layout()
# plt.show()
