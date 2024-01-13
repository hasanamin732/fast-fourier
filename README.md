# Image Processing with FFT and Gaussian Filtering

This Python script provides functionalities for efficient image processing using Fast Fourier Transform (FFT) and Gaussian filtering. The script includes three main functions:

## 1. `main`
- **Description:** Applies Gaussian filtering to an input image and displays the original and blurred images.
- **Arguments:**
  - `imgpath`: Path to the input image file.
  - `my_sigma`: Standard deviation for the Gaussian filter. Default is 25.
  - `main_low_pass`: Boolean indicating whether to apply a low-pass filter (True) or a high-pass filter (False).

## 2. `fft_visual`
- **Description:** Performs FFT on the input image and its Gaussian-filtered version, displaying the magnitude of the FFT.
- **Arguments:**
  - `imgpath`: Path to the input image file.
  - `my_sigma`: Standard deviation for the Gaussian filter. Default is 25.
  - `main_low_pass`: Boolean indicating whether to apply a low-pass filter (True) or a high-pass filter (False).

## 3. `fft_graph`
- **Description:** Extracts a row/column from the FFT images and plots the profiles of FFT peaks before and after filtering.
- **Arguments:**
  - `imgpath`: Path to the input image file.
  - `my_sigma`: Standard deviation for the Gaussian filter. Default is 25.
  - `main_low_pass`: Boolean indicating whether to apply a low-pass filter (True) or a high-pass filter (False).

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hasanamin732/fast-fourier
   cd fast-fourier
