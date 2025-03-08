
# Create your tests here.
import requests
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class APIClient(object):
    
    def __init__(self):
        self.BASE_URL = 'http://localhost:8000/api'

        self.endpoints = {
            'token':f'{self.BASE_URL}/token/',
            'get-routes':f'{self.BASE_URL}/get-routes/',
            'upload-image':f'{self.BASE_URL}/upload-image/',
            'get-grayscale':f'{self.BASE_URL}/get-grayscale/',
            'add-gaussian-noise':f'{self.BASE_URL}/add-gaussian-noise/',
            'add-uniform-noise':f'{self.BASE_URL}/add-uniform-noise/',
            'add-salt-and-pepper-noise':f'{self.BASE_URL}/add-salt-and-pepper-noise/',
            'blur':f'{self.BASE_URL}/blur/',
            'gaussian-blur':f'{self.BASE_URL}/gaussian-blur/',
            'median-blur':f'{self.BASE_URL}/median-blur/',
            'sobel-edge-detection':f'{self.BASE_URL}/sobel-edge-detection/',
            'roberts-edge-detection':f'{self.BASE_URL}/roberts-edge-detection/',
            'prewitt-edge-detection':f'{self.BASE_URL}/prewitt-edge-detection/',
            'canny-edge-detection':f'{self.BASE_URL}/canny-edge-detection/',
            'get-histogram':f'{self.BASE_URL}/get-histogram/',
            'get-equalized-histogram':f'{self.BASE_URL}/get-equalized-histogram/',
            'get-equalized-image':f'{self.BASE_URL}/get-equalized-image/',
            'normalize':f'{self.BASE_URL}/normalize/',
            'global-threshold':f'{self.BASE_URL}/global-threshold/',
            'local-threshold':f'{self.BASE_URL}/local-threshold/',
            'get-hybrid-image':f'{self.BASE_URL}/get-hybrid-image/',
            
        }

        self.headers = None

    def login(self, username,password):
        """
    Logs the user into the system using the provided credentials.

    Parameters:
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        bool: True if the login was successful and the access token was obtained, False otherwise.

    Raises:
        requests.exceptions.RequestException: If there is an error during the HTTP request.

    Usage:
        client = YourClientClass()
        username = "example_user"
        password = "example_password"
        if client.login(username, password):
            print("Login successful!")
        else:
            print("Login failed. Please check your credentials.")
    """ 
        res = requests.post(self.reverse('token'), data={
            'username': username,
            'password': password,
        })
        access_token = None
        if res.status_code == 200:
           access_token = res.json().get('access')
        else:
            return False

        self.headers = {
            'Authorization': 'Bearer '+ access_token
        }

        return True

    def upload_image(self,filename):
        """
    Uploads an image file to the server.

    Parameters:
        filename (str): The path to the image file to be uploaded.

    Raises:
        requests.exceptions.RequestException: If there is an error during the HTTP request.
    """
        self.files = {
            'image': open(filename,'rb')
        }

        res = requests.post(self.reverse('upload-image'), files=self.files, headers=self.headers)

    def reverse(self,endpoint):
        """
    Retrieves the URL associated with the specified endpoint.

    Parameters:
        endpoint (str): The endpoint for which the URL is to be retrieved.

    Returns:
        str: The URL associated with the specified endpoint.
    """
        
        return self.endpoints.get(endpoint)

    def get_grayscale(self):
        """
    Retrieves the grayscale version of an image from the server.

    Returns:
        QPixmap: A QPixmap object representing the grayscale image.
    """
        
        res = requests.get(self.reverse('get-grayscale'), headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]

        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
        

    def add_gaussian_noise(self,mean, std):
        """
    Adds Gaussian noise to an image retrieved from the server.

    Parameters:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.

    Returns:
        QPixmap: A QPixmap object representing the image with added Gaussian noise.
    """
        payload = {
            'mean': mean,
            'std': std,
        }
        res = requests.get(self.reverse('add-gaussian-noise'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def add_uniform_noise(self, low , high):
        """
    Adds uniform noise to an image retrieved from the server.

    Parameters:
        low (int): The lower bound of the uniform distribution.
        high (int): The upper bound of the uniform distribution.

    Returns:
        QPixmap: A QPixmap object representing the image with added uniform noise.
    """
        payload = {
            'low': low,
            'high':high
        }
        res = requests.get(self.reverse('add-uniform-noise'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def add_salt_and_pepper_noise(self, saltiness = 0.5, pepperiness = 0.5):
        """
    Adds salt and pepper noise to an image retrieved from the server.

    Parameters:
        saltiness (float): The probability of adding salt noise to each pixel.
        pepperiness (float): The probability of adding pepper noise to each pixel.

    Returns:
        QPixmap: A QPixmap object representing the image with added salt and pepper noise.
    """
     
        payload = {
            'saltiness': saltiness,
            'pepperiness': pepperiness
        }
        res = requests.get(self.reverse('add-salt-and-pepper-noise'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def blur(self, kernel_size = 3):
        """
    Applies a blur filter to an image retrieved from the server.

    Parameters:
        kernel_size (int): The size of the blur kernel.

    Returns:
        QPixmap: A QPixmap object representing the blurred image.
    """
        
        
        payload = {
            'kernel': kernel_size,

        }
        res = requests.get(self.reverse('blur'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def gaussian_blur(self, kernel_size = 3, std = 1):
        """
    Applies Gaussian blur to an image retrieved from the server.

    Parameters:
        kernel_size (int): The size of the Gaussian kernel.
        std (float): The standard deviation of the Gaussian distribution.

    Returns:
        QPixmap: A QPixmap object representing the image with Gaussian blur applied.
    """
        payload = {
            'kernel': kernel_size,
            'std':std,
        }
        res = requests.get(self.reverse('gaussian-blur'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def median_blur(self, kernel_size = 3):
        """
    Applies median blur to an image retrieved from the server.

    Parameters:
        kernel_size (int): The size of the kernel for median filtering.

    Returns:
        QPixmap: A QPixmap object representing the image with median blur applied.
    """
        
        payload = {
            'kernel': kernel_size,
        }
        res = requests.get(self.reverse('median-blur'),params=payload, headers=self.headers)

        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def sobel_edge_detection(self):
        """
    Performs Sobel edge detection on an image retrieved from the server.

    Returns:
        QPixmap: A QPixmap object representing the image after Sobel edge detection.
    """
        res = requests.get(self.reverse('sobel-edge-detection'), headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def roberts_edge_detection(self):
        """
    Performs Roberts edge detection on an image retrieved from the server.

    Returns:
        QPixmap: A QPixmap object representing the image after Roberts edge detection.
    """
        res = requests.get(self.reverse('roberts-edge-detection'), headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def prewitt_edge_detection(self):
        """
    Performs Prewitt edge detection on an image retrieved from the server.

    Returns:
        QPixmap: A QPixmap object representing the image after Prewitt edge detection.
    """
        res = requests.get(self.reverse('prewitt-edge-detection'), headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def canny_edge_detection(self, low_threshold = 50 , high_threshold = 150):
        """
    Performs Canny edge detection on an image retrieved from the server.

    Parameters:
        low_threshold (int): The lower threshold for the Canny edge detector.
        high_threshold (int): The upper threshold for the Canny edge detector.

    Returns:
        QPixmap: A QPixmap object representing the image after Canny edge detection.
    """
        payload = {
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
        }
        res = requests.get(self.reverse('canny-edge-detection'),params=payload, headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def get_histogram(self, channel = 0, image_type = 'gray'):
        """
    Retrieves the histogram of an image from the server.

    Parameters:
        channel (int): The channel for which to compute the histogram.
        image_type (str): The type of the image ('gray' or 'color').

    Returns:
        QPixmap: A QPixmap object representing the histogram of the image.
    """
        payload = {
            'channel': channel,
            'image_type': image_type,
        }
        res = requests.get(self.reverse('get-histogram'),params=payload, headers=self.headers)
        
        histogram = np.frombuffer(res.content, dtype=np.uint8)
        
        plt.hist(histogram, bins=256, color='blue')
        plt.xlabel('levels')
        plt.ylabel('Frequency')
        plt.title('Basic Histogram')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0) 
        image_bytes = buffer.getvalue()
        qimage = QImage.fromData(image_bytes)
        pixmap = QPixmap.fromImage(qimage)
        plt.close()
        return pixmap
        
        

    def get_equalized_histogram(self):
        """
    Retrieves the equalized histogram of an image from the server.

    Returns:
        QPixmap: A QPixmap object representing the equalized histogram of the image.
    """
        
        res = requests.get(self.reverse('get-equalized-histogram'), headers=self.headers)
        
        equalized_histogram = np.frombuffer(res.content, dtype=np.uint8)
        plt.hist(equalized_histogram, bins=256, color='blue')
        plt.xlabel('levels')
        plt.ylabel('Frequency')
        plt.title('Equalized Histogram')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0) 
        image_bytes = buffer.getvalue()
        qimage = QImage.fromData(image_bytes)
        pixmap = QPixmap.fromImage(qimage)
        plt.close()
        return pixmap
    
    def normalize(self):
        pass
        # res = requests.get(self.reverse('normalize'), headers=self.headers)
        # index = res.headers.get('Content-Type').find('/')+1
        # out_file = 'output.'+res.headers.get('Content-Type')[index:]
        # img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        # image =cv.imdecode(img_bytes, cv.IMREAD_COLOR)
        # cv.imshow(out_file,image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
    
    def global_threshold(self, threshold = 50):
        """
    Applies global thresholding to an image retrieved from the server.

    Parameters:
        threshold (int): The threshold value for global thresholding.

    Returns:
        QPixmap: A QPixmap object representing the image after global thresholding.
    """
        payload = {
            'threshold': threshold,
        }
        res = requests.get(self.reverse('global-threshold'),params=payload, headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        image =cv.imdecode(img_bytes, cv.IMREAD_COLOR)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def local_threshold(self, kernel_size = 11):
        """
    Applies local thresholding to an image retrieved from the server.

    Parameters:
        kernel_size (int): The size of the kernel for local thresholding.

    Returns:
        QPixmap: A QPixmap object representing the image after local thresholding.
    """
        payload = {
            'kernel': kernel_size,
        }
        res = requests.get(self.reverse('local-threshold'),params=payload, headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def get_equalized_image(self):
        """
    Retrieves the equalized version of an image from the server.

    Returns:
        QPixmap: A QPixmap object representing the equalized image.
    """
        
        res = requests.get(self.reverse('get-equalized-image'), headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def get_hybrid_image(self,filename_1,filename_2,low_pass_cuttoff_freq, high_pass_cuttoff_freq ):
        """
    Retrieves a hybrid image generated from two input images.

    Parameters:
        filename_1 (str): The path to the first input image file.
        filename_2 (str): The path to the second input image file.
        low_pass_cuttoff_freq (float): The cutoff frequency for the low-pass filter.
        high_pass_cuttoff_freq (float): The cutoff frequency for the high-pass filter.

    Returns:
        QPixmap: A QPixmap object representing the hybrid image.
    """
        data = {
            'low_pass_cuttoff_freq':low_pass_cuttoff_freq,
            'high_pass_cuttoff_freq':high_pass_cuttoff_freq
        }
        files = {
            'first_image': open(filename_1,'rb'),
            'second_image': open(filename_2,'rb'),
        }
        res = requests.post(self.reverse('get-hybrid-image'),data=data,files = files, headers=self.headers)
        index = res.headers.get('Content-Type').find('/')+1
        out_file = 'output.'+res.headers.get('Content-Type')[index:]
        img_bytes = np.frombuffer(res.content, dtype=np.uint8)
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    def get_low_pass_filter(self, low_pass_cuttoff_freq,shape = (500,500)):
        rows_1, cols_1 = shape
        # center the image
        crow_1, ccol_1 = rows_1 // 2, cols_1 // 2
        low_pass_filter = np.zeros((rows_1, cols_1), np.uint8)
        low_pass_filter[crow_1 - low_pass_cuttoff_freq:crow_1 + low_pass_cuttoff_freq, ccol_1 - low_pass_cuttoff_freq:ccol_1 + low_pass_cuttoff_freq] = 1
        img_back = np.fft.ifft2(low_pass_filter)
        img_back = np.abs(img_back)
        img_bytes = img_back.astype(np.uint8).tobytes()
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap
    def get_high_pass_filter(self, high_pass_cuttoff_freq,shape = (500,500)):
        rows_2, cols_2 = shape
        # center the image
        crow_2, ccol_2 = rows_2 // 2, cols_2 // 2
        high_pass_filter = np.ones((rows_2, cols_2), np.uint8)
        high_pass_filter[crow_2 - high_pass_cuttoff_freq:crow_2 + high_pass_cuttoff_freq, ccol_2 - high_pass_cuttoff_freq:ccol_2 + high_pass_cuttoff_freq] = 0
        img_back = np.fft.ifft2(high_pass_filter)
        img_back = np.abs(img_back)
        img_bytes = img_back.astype(np.uint8).tobytes()
        qimage = QImage.fromData(img_bytes)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap




