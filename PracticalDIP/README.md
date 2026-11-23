# Visual Outputs

## Original Image

The original image below was supplied for this run and used as the input for all 15 practicals. The scripts resize it to 256×256 before processing.

![Original Iron Man Image](images/IRON.jpeg)

### 01. 3x3 Mean Filter
- **Test matrix center:** `22.2222` (`200 / 9`) · **Gaussian noise (σ = 0.05) PSNR:** `26.67 dB` → `27.23 dB`

![01 Mean Filter](outputs/01_mean_filter.png)

### 02. 3x3 Weighted Average Filter
- **Test matrix center:** `28.1250` (`450 / 16`) · **Gaussian noise (σ = 0.05) PSNR:** `26.67 dB` → `28.66 dB`

![02 Weighted Average Filter](outputs/02_weighted_average_filter.png)

### 03. 5x5 Gaussian Spatial Filter (σ = 1.0)
- **5x5 Gaussian kernel sum:** `1.0` · **Gaussian noise (σ = 0.05) PSNR:** `26.67 dB` → `27.27 dB`

![03 Gaussian Filter](outputs/03_gaussian_filter.png)

### 04. 3x3 Median Filter
- **Test matrix center:** `20.0000` · **Salt-and-pepper noise PSNR:** `17.36 dB` → `29.12 dB`

![04 Median Filter](outputs/04_median_filter.png)

### 05. 3x3 Laplacian Sharpening Filter

![05 Laplacian Filter](outputs/05_laplacian_filter.png)

### 06. Sobel Edge Detection Filter

![06 Sobel Filter](outputs/06_sobel_filter.png)

### 07. High-Boost Sharpening Filter (A = 1.5)

![07 High Boost Filter](outputs/07_high_boost_filter.png)

### 08. Ideal Low-Pass Filter (ILPF, D₀ = 40)

![08 Ideal Low Pass Filter](outputs/08_ideal_low_pass_filter.png)

### 09. Butterworth Low-Pass Filter (BLPF, n = 2, D₀ = 40)

![09 Butterworth Low Pass Filter](outputs/09_butterworth_low_pass_filter.png)

### 10. Gaussian Low-Pass Filter (GLPF, D₀ = 40)

![10 Gaussian Low Pass Filter](outputs/10_gaussian_low_pass_filter.png)

### 11. Ideal High-Pass Filter (IHPF, D₀ = 40)

![11 Ideal High Pass Filter](outputs/11_ideal_high_pass_filter.png)

### 12. Butterworth High-Pass Filter (BHPF, n = 2, D₀ = 40)

![12 Butterworth High Pass Filter](outputs/12_butterworth_high_pass_filter.png)

### 13. Gaussian High-Pass Filter (GHPF, D₀ = 40)

![13 Gaussian High Pass Filter](outputs/13_gaussian_high_pass_filter.png)

### 14. Huffman Coding (8-bit Grayscale)
- **Image (256×256):** `524,288` bits → `453,203` encoded bits (`1.157x`, lossless `True`)

![14 Huffman Coding](outputs/14_huffman_coding.png)

### 15. Arithmetic Coding (4-bit Quantized Grayscale)
- **Image (256×256):** `262,144` bits → `226,769` estimated encoded bits (`1.156x`, lossless `True`)

![15 Arithmetic Coding](outputs/15_arithmetic_coding.png)
