from Gaussian import *
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def main(imgpath:str,my_sigma:float,main_low_pass:bool):
    image = plt.imread(imgpath)  # Read the image

    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    
    if my_sigma==0:
        my_sigma=10
        print("Zero is not allowed for sigma, running on default value 10")

    # Apply the Gaussian low-pass filter
    
    filtered_image=gaussian_filter(image,sigma=my_sigma,low_pass=main_low_pass)
    # Display original and filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'Blurred Image with $\sigma$={my_sigma}')
    plt.tight_layout()
    plt.show()

def fft_visual(imgpath:str,my_sigma:float,main_low_pass:bool):
    image = plt.imread(imgpath)  # Read the image

    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    if my_sigma==0:
        my_sigma=10
        print("Zero is not allowed for sigma, running on default value 10")
    # Apply the Gaussian low-pass filter
    
    filtered_image_fft,image_fft=gaussian_filter(image,sigma=my_sigma,low_pass=main_low_pass,ifft=False)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(np.log(np.abs(image_fft.array) + 1), cmap='gray')  # Log magnitude of original image FFT
    plt.title('Original Image FFT Magnitude')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(filtered_image_fft) + 1), cmap='gray')  # Log magnitude of filtered image FFT
    plt.title(f'Filtered Image FFT Magnitude with $\sigma$={my_sigma}')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def fft_graph(imgpath:str,my_sigma:float,main_low_pass:bool):
    image = plt.imread(imgpath)  # Read the image

    # Convert to grayscale if it's a color image
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    if my_sigma==0:
        my_sigma=10
        print("Zero is not allowed for sigma, running on default value 10")
    # Apply the Gaussian low-pass filter
    
    filtered_image_fft,image_fft=gaussian_filter(image,sigma=my_sigma,low_pass=main_low_pass,ifft=False)
    
    # Extract a row/column from the FFT images to visualize the peaks before and after filtering
    row_to_visualize = 100  # You can choose a row or column to visualize
    fft_row_original = np.abs(image_fft.array[row_to_visualize, :])
    fft_row_filtered = np.abs(filtered_image_fft[row_to_visualize, :])

    # Plotting the profiles of the FFT peaks before and after filtering
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(fft_row_original)), fft_row_original, label='Original FFT')
    plt.plot(np.arange(len(fft_row_filtered)), fft_row_filtered, label='Filtered FFT')
    plt.title(f'1D Profile of FFT Peaks (Row {row_to_visualize}), $\sigma$={my_sigma}')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__=="__main__":
    main('kingfisher.jpg',15,True)

