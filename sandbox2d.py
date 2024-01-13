import numpy as np
import timeit
import matplotlib.pyplot as plt
from fft import FFT2D  # Assuming you have FFT2D implemented in the fft module

def dft(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

def fft(x):
    x_obj = FFT2D(x)
    return x_obj._fft1d(x)

def measure_time_efficiency(func, array_lengths):
    execution_times = []

    for length in array_lengths:
        x = np.random.random(length)  # Generate a random array of given length

        # Measure the execution time
        time_taken = timeit.timeit(lambda: func(x), number=10) / 10  # Taking the average of 10 runs
        execution_times.append(time_taken)

    return execution_times

def plot_results(array_lengths, dft_times, fft_times):
    plt.plot(array_lengths, dft_times, label='DFT')
    plt.plot(array_lengths, fft_times, label='FFT')
    plt.xlabel('Array Length')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.title('Time Efficiency Comparison: DFT vs FFT')
    plt.show()

if __name__ == "__main__":
    array_lengths = 2**np.arange(5, 10)  # Add more lengths as needed

    dft_times = measure_time_efficiency(dft, array_lengths)
    fft_times = measure_time_efficiency(fft, array_lengths)

    plot_results(array_lengths, dft_times, fft_times)
