from matplotlib import image, pyplot as plt
import numpy as np
from fft import FFT2D

# Load image as pixel array
img = image.imread('kingfisher.jpg',)

gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

obj=FFT2D(gray_img)
img_fft=obj.fft2d()
img_ifft=img_fft.ifft2d(np.shape(gray_img))

res=gray_img-img_ifft

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(img_ifft)
plt.subplot(1,2,2)
plt.imshow(gray_img)
# plt.title("Error")

plt.tight_layout()
# plt.colorbar()
plt.show()
