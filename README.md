# VisionPro: Intelligent Image Processing Tool

## Overview

VisionPro is a **PyQt-based** interactive image processing application designed to provide users with a **wide range of image processing functionalities**. The application supports **image filtering, noise addition, edge detection, thresholding, histogram analysis, and hybrid image generation**. It is ideal for **image analysis, computer vision research, and educational purposes**.

## Features

- **Image Loading & Display**: Supports multiple image formats (PNG, JPG, BMP, TIFF, etc.)
- **Noise Addition**: Gaussian, uniform, and salt-and-pepper noise.
- **Blur Filters**: Median blur, average blur, and Gaussian blur.
- **Edge Detection**: Sobel, Prewitt, and Roberts edge detection.
- **Thresholding**: Global and local thresholding for segmentation.
- **Histogram Analysis**: Histogram visualization and equalization.
- **Hybrid Image Generation**: Combines two images with customizable frequency cutoffs.
- **Backend API Integration**: Processes images via a connected backend service.
- **User Authentication**: Secure login system for backend access.

## Image Processing Techniques

### 1. Edge Detection

- **Roberts Operator**: Detects edges without preprocessing, highlighting intensity changes.
- **Sobel & Prewitt Operators**: More refined edge detection with noise handling.

### 2. Image Filtering & Noise Handling

- **Gaussian Blur (Kernel=7, STD=10)**: Smooths images while preserving structural details.
- **Gaussian Noise (Mean=20, STD=30)**: Simulates real-world imaging noise for testing algorithms.

### 3. Grayscale Transformation

- Converts images to grayscale for simplified processing while preserving intensity variations.

### 4. Hybrid Image Generation

- Merges two images with specified **low-pass and high-pass filters**, enhancing important features.

## Installation

To set up and run the application, install the dependencies:

```bash
pip install PyQt5 opencv-python numpy matplotlib
```

## Usage

Run the application using:

```bash
python main.py
```

## Future Enhancements

- **Additional image processing algorithms and filters.**
- **Integration of advanced edge detection methods (e.g., Canny, Laplacian).**
- **Enhanced UI with interactive visualization tools.**

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

