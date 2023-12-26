import matplotlib.pyplot as plt
import numpy as np
from fft import FFT2D

data = np.array([
        [1, 2, 3,10],
        [4, 5, 6,11],
        [7, 8, 9,12],
        [13,14,15,16]
    ])

obj=FFT2D(data)
res=np.fft.fft2(data)
res1=np.fft.ifft2(res)
res4=obj.fft2d()
# res2=obj.fft2d().ifft2d()
res2=res4.ifft2d()
res3=res1-res2
print(res2)
plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(res1.real, cmap='viridis')
# plt.title('NumPy FFT Magnitude')

# plt.subplot(1, 3, 2)
# plt.imshow(res2.real, cmap='viridis')
# plt.title('Custom FFT Magnitude')

# plt.subplot(1, 3, 3)
plt.imshow(res3.real,cmap='Oranges_r')
plt.title("Error")

plt.tight_layout()
plt.colorbar()
plt.show()
