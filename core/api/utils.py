import cv2 as cv
import numpy as np


def gaussian_kernel(size, std):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*std**2)) * np.exp(-((x - size//2)**2 + (y - size//2)**2)/(2*std**2)),
        (size, size))
    kernel /= np.sum(kernel)
    return kernel


def apply_median_blur(image, kernel_size):
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    blurred_image = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            blurred_image[i, j] = np.median(neighborhood)
    
    return blurred_image.astype(np.uint8)

def apply_sobel_edge_detection(gray_image):
   
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = cv.filter2D(src = gray_image,kernel=sobel_x,anchor=(-1,-1), ddepth = -1)
    gradient_y = cv.filter2D(src = gray_image,kernel=sobel_y,anchor=(-1,-1), ddepth = -1)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude *= 255.0 / np.max(magnitude)
    
    return magnitude.astype(np.uint8)


def apply_roberts_edge_detection(gray_image):
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    gradient_x = cv.filter2D(gray_image, kernel = roberts_x,anchor=(-1,-1), ddepth = -1)
    gradient_y = cv.filter2D(gray_image, kernel = roberts_y,anchor=(-1,-1), ddepth = -1)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude *= 255.0 / np.max(magnitude)
    
    return magnitude.astype(np.uint8)

def apply_prewitt_edge_detection(gray_image):
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    gradient_x = cv.filter2D(gray_image, kernel = prewitt_x,anchor=(-1,-1), ddepth = -1)
    gradient_y = cv.filter2D(gray_image, kernel = prewitt_y,anchor=(-1,-1), ddepth = -1)
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude *= 255.0 / np.max(magnitude)
    
    return magnitude.astype(np.uint8)
    
def rgb_to_gray(pixel):
    # Convert RGB pixel to grayscale using luminosity method
    # Gray = 0.299 * R + 0.587 * G + 0.114 * B
    return int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])