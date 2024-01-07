import matplotlib.pyplot as plt
import numpy as np
from fft import FFT2D

data = np.array([
        [1, 2, 3,10],
        [4, 5, 6,11],
        [7, 8, 9,12],
        [13,14,15,16],
        [13,14,15,16]
    ])

res=np.fft.fft2(data)
res1=np.fft.ifft2(res)

obj=FFT2D(data)
res4=obj.fft2d()
# res4=FFT2D(res4)
# res2=obj.fft2d().ifft2d(data.shape)
res2=res4.ifft2d(data.shape)
res3=res1-res2
# res3=res-res4.array
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
