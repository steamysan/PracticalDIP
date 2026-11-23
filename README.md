# Digital Image Processing (DIP) Practicals

This repository contains 15 Digital Image Processing practical programs in the same numbered sequence as the reference repository. Each program is self-contained and saves its visual result to the `outputs/` directory.

## Sample Image

The examples use a different sample image (a cat photograph from `scikit-image`) instead of the reference image.

![Sample image](images/sample_cat.png)

## Requirements and Usage

```bash
pip install -r requirements.txt
python 01_mean_filter.py
```

Run each numbered Python file from the repository root. Outputs are saved automatically in `outputs/`. The programs use `images/sample_cat.png`; replace that file or update `IMAGE_FILE` in a script to use another image.

## Practical Results

### 01. 3x3 Mean Filter

Spatial smoothing using a uniform averaging kernel.

![3x3 Mean Filter](outputs/01_mean_filter.jpg)

### 02. 3x3 Weighted Average Filter

Spatial smoothing with greater weight at the center pixel.

![3x3 Weighted Average Filter](outputs/02_weighted_average_filter.jpg)

### 03. 5x5 Gaussian Spatial Filter

Gaussian smoothing with sigma = 1.

![5x5 Gaussian Spatial Filter](outputs/03_gaussian_filter.jpg)

### 04. 3x3 Median Filter

Nonlinear spatial filter effective against salt-and-pepper noise.

![3x3 Median Filter](outputs/04_median_filter.jpg)

### 05. Laplacian Sharpening Filter

Sharpening using the second spatial derivative.

![Laplacian Sharpening Filter](outputs/05_laplacian_filter.jpg)

### 06. Sobel Edge Detection Filter

Edge detection using horizontal and vertical gradients.

![Sobel Edge Detection Filter](outputs/06_sobel_filter.jpg)

### 07. High-Boost Sharpening Filter

Detail enhancement using the original image and a blurred mask.

![High-Boost Sharpening Filter](outputs/07_high_boost_filter.jpg)

### 08. Ideal Low-Pass Filter

Frequency-domain low-pass filtering with a sharp cut-off.

![Ideal Low-Pass Filter](outputs/08_ideal_low_pass_filter.jpg)

### 09. Butterworth Low-Pass Filter

Frequency-domain low-pass filtering with order n = 2.

![Butterworth Low-Pass Filter](outputs/09_butterworth_low_pass_filter.jpg)

### 10. Gaussian Low-Pass Filter

Smooth frequency-domain low-pass response.

![Gaussian Low-Pass Filter](outputs/10_gaussian_low_pass_filter.jpg)

### 11. Ideal High-Pass Filter

Frequency-domain high-pass filtering with a sharp cut-off.

![Ideal High-Pass Filter](outputs/11_ideal_high_pass_filter.jpg)

### 12. Butterworth High-Pass Filter

Frequency-domain high-pass filtering with order n = 2.

![Butterworth High-Pass Filter](outputs/12_butterworth_high_pass_filter.jpg)

### 13. Gaussian High-Pass Filter

Smooth frequency-domain high-pass response.

![Gaussian High-Pass Filter](outputs/13_gaussian_high_pass_filter.jpg)

### 14. Huffman Coding

Lossless grayscale image coding with variable-length prefix codes.

![Huffman Coding](outputs/14_huffman_coding.jpg)

### 15. Arithmetic Coding

Lossless arithmetic coding of 4-bit quantized grayscale blocks.

![Arithmetic Coding](outputs/15_arithmetic_coding.jpg)

## Practical List

01. 3x3 Mean Filter — `01_mean_filter.py`
02. 3x3 Weighted Average Filter — `02_weighted_average_filter.py`
03. 5x5 Gaussian Spatial Filter — `03_gaussian_filter.py`
04. 3x3 Median Filter — `04_median_filter.py`
05. Laplacian Sharpening Filter — `05_laplacian_filter.py`
06. Sobel Edge Detection Filter — `06_sobel_filter.py`
07. High-Boost Sharpening Filter — `07_high_boost_filter.py`
08. Ideal Low-Pass Filter — `08_ideal_low_pass_filter.py`
09. Butterworth Low-Pass Filter — `09_butterworth_low_pass_filter.py`
10. Gaussian Low-Pass Filter — `10_gaussian_low_pass_filter.py`
11. Ideal High-Pass Filter — `11_ideal_high_pass_filter.py`
12. Butterworth High-Pass Filter — `12_butterworth_high_pass_filter.py`
13. Gaussian High-Pass Filter — `13_gaussian_high_pass_filter.py`
14. Huffman Coding — `14_huffman_coding.py`
15. Arithmetic Coding — `15_arithmetic_coding.py`
