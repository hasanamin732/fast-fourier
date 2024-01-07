import numpy as np
import matplotlib.pyplot as plt
from fft import FFT2D

N = int(2**10)
x = np.linspace(0,10,N)
y = np.sin(8*x)+np.sin(11*x)

obj = FFT2D(y)

res2 = obj.fft1d(y)
res=np.fft.fft(y)
res_real = np.abs(res)
res_real2 = np.abs(res2)
# print(res_real2==res_real)

comparison = res_real2 == res_real

# Calculate the percentage of True values
percentage_equal = np.count_nonzero(comparison) / len(comparison) * 100

print(f"Percentage of equal values: {percentage_equal:.2f}%")
print(np.argwhere(res_real > 349))
print(np.argwhere(res_real2 > 349))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plotting the FFT results without vertical lines
ax[0].plot(np.arange(N)[:N // 2], res_real2[:N // 2], label='Custom FFT',color="green")
ax[0].plot(np.arange(N)[:N // 2], res_real[:N // 2], label='NumPy FFT')
ax[0].legend()
ax[0].set_title('FFT Results')
ax[0].set_xlabel('Frequency')

# Plotting the same FFT results with vertical lines on the second subplot
ax[1].plot(np.arange(N)[:N // 2], res_real[:N // 2], label='NumPy FFT')
ax[1].plot(np.arange(N)[:N // 2], res_real2[:N // 2], label='Custom FFT')
# ax[1].axvline(1 * (np.pi / 2), color='red', linestyle='--', label='Vertical line 1')
# ax[1].axvline(7 * (np.pi / 2), color='green', linestyle='--', label='Vertical line 2')
# ax[1].axvline(8 * (np.pi / 2), color='blue', linestyle='--', label='Vertical line 3')
ax[1].legend()
ax[1].set_title('FFT Results with Vertical Lines')
ax[1].set_xlabel('Frequency')

plt.tight_layout()
plt.show()
