from Gaussian import *
from matplotlib import image, pyplot as plt
import numpy as np

# # Load image as pixel array
# img = image.imread('kingfisher_gray.jpg')


# data = img.mean(axis=2) if img.ndim == 3 else img
# # print(data.shape)
# obj=FFT2D(data)

# res4=obj.fft2d()

# # res2=obj.fft2d().ifft2d()
# kernal=gaussian_kernel(res4.shape,1)
# convolved=convolve2d(res4,kernal,mode='same', boundary='symm')
# obj2=FFT2D(convolved)
# res2=obj2.ifft2d(data.shape)
# # print(res2)
# plt.figure(figsize=(10, 5))
# plt.imshow(res2)
# plt.show()

# Load an example image (replace 'image_path' with your image path)
image = plt.imread('rose_gray.jpg')  # Read the image

# Convert to grayscale if it's a color image
if image.ndim == 3:
    image = np.mean(image, axis=2)

# Apply the Gaussian low-pass filter
filtered_image = gaussian_low_pass_filter(image, sigma=0.5)

# Display original and filtered images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')

plt.tight_layout()
plt.show()
