from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import MyTokenObtainPairSerializer
from django.http import HttpResponse
from ..models import UserImage, HybridImageComponents
from rest_framework import status
import cv2 as cv
import numpy as np
import os
from . import utils
import pandas as pd
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_routes(request):
    routes = [
        '/api/get-routes/',
        '/api/',
    ]
    return Response(routes)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def upload_image(request):
    images = UserImage.objects.filter(user=request.user)
    if images.count() > 0:
        for image in images:
            image.delete()
    instance = UserImage(user=request.user, image=request.FILES['image'], out_image = request.FILES['image'])
    instance.save()
    return Response(status = status.HTTP_201_CREATED)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_grayscale(request):
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        image = cv.imread(filename)
        gray_image  = np.zeros(shape = (image.shape[0], image.shape[1]), dtype = np.uint8)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                R = image[x,y,0]
                G = image[x,y,1]
                B = image[x,y,2]
                gray_image[x,y] = utils.rgb_to_gray((R,G,B))

        cv.imwrite(filename, gray_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def add_gaussian_noise(request):
    mean = request.GET.get('mean',None)
    std = request.GET.get('std',None)
    if std is None or mean is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    mean = float(mean)
    std = float(std)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        row,col = gray_image.shape
        noise = np.random.normal(mean,std, (row,col))
        noisy_image = gray_image + noise
        noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)
        cv.imwrite(filename, noisy_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def add_uniform_noise(request):
    low = request.GET.get('low',None)
    high = request.GET.get('high',None)
    if low is None or high is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    low = float(low)
    high = float(high)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        row,col = gray_image.shape
        noise = np.random.uniform(low,high, (row,col))
        noisy_image = gray_image + noise
        noisy_image = np.clip(noisy_image,0,255).astype(np.uint8)
        cv.imwrite(filename, noisy_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def add_salt_and_pepper_noise(request):
    saltiness = request.GET.get('saltiness',None)
    pepperiness = request.GET.get('pepperiness',None)
    if saltiness is None and pepperiness is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        noisy_image = np.copy(gray_image)
        width,height = gray_image.shape
        if saltiness is not None:
            saltiness = float(saltiness)
            num_salt = np.floor(gray_image.size * saltiness)
            salt_x_vector = np.random.randint(0,width, int(num_salt))
            salt_y_vector = np.random.randint(0,height, int(num_salt))
            noisy_image[salt_x_vector, salt_y_vector] = 255
        if pepperiness is not None:
            pepperiness = float(pepperiness)
            num_pepper = np.floor(gray_image.size * pepperiness)
            pepper_x_vector = np.random.randint(0,width, int(num_pepper))
            pepper_y_vector = np.random.randint(0,height, int(num_pepper))
            noisy_image[pepper_x_vector, pepper_y_vector] = 0

        cv.imwrite(filename, noisy_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def blur(request):
    kernel_size = request.GET.get('kernel',None)
    if kernel_size is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    kernel_size = int(kernel_size)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        row,col = gray_image.shape
        kernel = np.ones(shape = (kernel_size, kernel_size) ) / (kernel_size**2)
        filtered_image = cv.filter2D(src = gray_image,kernel=kernel,anchor=(-1,-1), ddepth = -1)
        cv.imwrite(filename, filtered_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def gaussian_blur(request):
    kernel_size = request.GET.get('kernel',None)
    std = request.GET.get('std',None)
    if kernel_size is None or std is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    kernel_size = int(kernel_size)
    std = float(std)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        row,col = gray_image.shape
        kernel = utils.gaussian_kernel(kernel_size, std)
        filtered_image = cv.filter2D(src = gray_image,kernel=kernel,anchor=(-1,-1), ddepth = -1)
        cv.imwrite(filename, filtered_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def median_blur(request):
    kernel_size = request.GET.get('kernel',None)
    if kernel_size is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    kernel_size = int(kernel_size)
    try:
        user_image = UserImage.objects.get(user = request.user)
        
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        filtered_image = utils.apply_median_blur(gray_image, kernel_size)
        cv.imwrite(filename, filtered_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def sobel_edge_detection(request):
    
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edited_image = utils.apply_sobel_edge_detection(gray_image)
        cv.imwrite(filename, edited_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def roberts_edge_detection(request):
    
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edited_image = utils.apply_roberts_edge_detection(gray_image)
        cv.imwrite(filename, edited_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def prewitt_edge_detection(request):
    
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edited_image = utils.apply_prewitt_edge_detection(gray_image)
        cv.imwrite(filename, edited_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def canny_edge_detection(request):
    low_threshold = request.GET.get('low_threshold',None)
    high_threshold = request.GET.get('high_threshold',None)
    if low_threshold is None or high_threshold is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    low_threshold = int(low_threshold)
    high_threshold = int(high_threshold)
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edited_image = cv.Canny(gray_image, low_threshold, high_threshold)
        cv.imwrite(filename, edited_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_histogram(request):
    channel = request.GET.get('channel',None)
    image_type = request.GET.get('image_type',None)
    if channel is None or image_type is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    channel = int(channel)
    image_type = str(image_type)

    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        N_LEVELS = 256
        histogram = np.zeros(N_LEVELS, dtype=np.uint8)
        if image_type == 'gray':
            for row in gray_image:
                for pixel in row:
                    histogram[pixel] += 1
        elif image_type == 'rgb':
            for row in img:
                for pixel in row:
                    histogram[pixel[channel]] += 1
        histogram_bytes = histogram.tobytes() 
        return HttpResponse(histogram_bytes)
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_equalized_histogram(request):
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        N_LEVELS = 256
        histogram = np.zeros(N_LEVELS, dtype=np.uint8)
        for row in gray_image:
            for pixel in row:
                histogram[pixel] += 1
        pdf_vector = histogram / np.sum(histogram)
        cdf_vector = np.cumsum(histogram) / np.sum(histogram)
        max_level = np.max(gray_image)
        scaled_histogram = cdf_vector * max_level
        equalized_histogram = np.round(scaled_histogram).astype(np.uint8).tobytes()
        return HttpResponse(equalized_histogram)
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_equalized_image(request):
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        N_LEVELS = 256
        histogram = np.zeros(N_LEVELS, dtype=np.uint8)
        for row in gray_image:
            for pixel in row:
                histogram[pixel] += 1
        pdf_vector = histogram / np.sum(histogram)
        cdf_vector = np.cumsum(histogram) / np.sum(histogram)
        max_level = np.max(gray_image)
        scaled_histogram = cdf_vector * max_level
        equalized_histogram = np.round(scaled_histogram)
        equalized_image = np.copy(gray_image)
        for i in range(len(gray_image)):
            for j in range(len(gray_image[i])):
                equalized_image[i][j]= equalized_histogram[gray_image[i][j]]
        equalized_image = equalized_image.astype(np.uint8)
        cv.imwrite(filename, equalized_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def normalize(request):
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_float = gray_image.astype(np.float32)
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        normalized_image = (image_float - min_val) / (max_val - min_val)
        cv.imwrite(filename, normalized_image)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def global_threshold(request):

    threshold = request.GET.get('threshold',None)
    if threshold is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    threshold = int(threshold)
    
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, global_thresholded = cv.threshold(gray_image, threshold, 255, cv.THRESH_BINARY)
        cv.imwrite(filename, global_thresholded)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def local_threshold(request):
    kernel_size = request.GET.get('kernel',None)
    if kernel_size is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    kernel_size = int(kernel_size)
    try:
        user_image = UserImage.objects.get(user = request.user)
        filename = str(user_image.out_image)
        img = cv.imread(filename)
        gray_image  = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        SUBTRACTED_FROM_MEAN = 2
        local_thresholded = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernel_size, SUBTRACTED_FROM_MEAN)
        cv.imwrite(filename, local_thresholded)
        user_image.save()
        with open(filename, 'rb') as f:
            extension = os.path.splitext(filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except UserImage.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_hybrid_image(request):
    low_pass_cuttoff_freq = request.POST.get('low_pass_cuttoff_freq',None)
    high_pass_cuttoff_freq = request.POST.get('high_pass_cuttoff_freq',None)
    first_image = request.FILES.get('first_image')
    second_image = request.FILES.get('second_image')
    images = HybridImageComponents.objects.filter(user=request.user)
    if images.count() > 0:
        for image in images:
            image.delete()
    instance = HybridImageComponents(user=request.user, first_image = first_image, second_image = second_image)
    instance.save()

    if low_pass_cuttoff_freq is None or \
        high_pass_cuttoff_freq is None:
        return Response(status = status.HTTP_400_BAD_REQUEST)
    low_pass_cuttoff_freq = int(low_pass_cuttoff_freq)
    high_pass_cuttoff_freq = int(high_pass_cuttoff_freq)
    try:
        hybrid_image = HybridImageComponents.objects.get(user = request.user)
        first_filename = str(hybrid_image.first_image)
        second_filename = str(hybrid_image.second_image)
        first_image = cv.cvtColor(cv.imread(first_filename),cv.COLOR_BGR2GRAY)
        second_image = cv.cvtColor(cv.imread(second_filename),cv.COLOR_BGR2GRAY)
        first_image = cv.resize(first_image,(300,400))
        second_image = cv.resize(second_image,(300,400))
        f_transform_1 = np.fft.fft2(first_image)
        f_shift_1 = np.fft.fftshift(f_transform_1)
        f_transform_2 = np.fft.fft2(second_image)
        f_shift_2 = np.fft.fftshift(f_transform_2)

        rows_1, cols_1 = first_image.shape
        # center the image
        crow_1, ccol_1 = rows_1 // 2, cols_1 // 2
        low_pass_filter = np.zeros((rows_1, cols_1), np.uint8)
        low_pass_filter[crow_1 - low_pass_cuttoff_freq:crow_1 + low_pass_cuttoff_freq, ccol_1 - low_pass_cuttoff_freq:ccol_1 + low_pass_cuttoff_freq] = 1
        f_shift_filtered_1 = f_shift_1 * low_pass_filter

        rows_2, cols_2 = second_image.shape
        # center the image
        crow_2, ccol_2 = rows_2 // 2, cols_2 // 2
        high_pass_filter = np.ones((rows_2, cols_2), np.uint8)
        high_pass_filter[crow_2 - high_pass_cuttoff_freq:crow_2 + high_pass_cuttoff_freq, ccol_2 - high_pass_cuttoff_freq:ccol_2 + high_pass_cuttoff_freq] = 0
        f_shift_filtered_2 = f_shift_2 * high_pass_filter
        f_shift_filtered = f_shift_filtered_1 + f_shift_filtered_2

        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        out_filename = 'images/hybrid_' + hybrid_image.user.username+'.jpg'
        cv.imwrite(out_filename, img_back)
        with open(out_filename, 'rb') as f:
            extension = os.path.splitext(out_filename)[1] 
            return HttpResponse(f, content_type='image/'+ extension[1:])
    except HybridImageComponents.DoesNotExist:
        return Response(status = status.HTTP_404_NOT_FOUND)

    return Response(status = status.HTTP_200_OK)