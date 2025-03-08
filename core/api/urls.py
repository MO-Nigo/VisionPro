from django.urls import path
from . import views

from rest_framework_simplejwt.views import TokenRefreshView
from .views import MyTokenObtainPairView

urlpatterns = [
    path('get-routes/', views.get_routes, name = 'get-routes'),
    path('token/', MyTokenObtainPairView.as_view(), name='token-obtain-pair'),
    path('token/refresh/', TokenRefreshView.as_view(), name='token-refresh'),
    path('upload-image/', views.upload_image, name = 'upload-image'),
    path('get-grayscale/', views.get_grayscale , name = 'get-grayscale'),
    path('add-gaussian-noise/', views.add_gaussian_noise, name = 'add-gaussian-noise'),
    path('add-uniform-noise/', views.add_uniform_noise, name = 'add-uniform-noise'),
    path('add-salt-and-pepper-noise/', views.add_salt_and_pepper_noise,  name = 'add-salt-and-pepper-noise'),
    path('blur/', views.blur, name = 'blur'),
    path('gaussian-blur/', views.gaussian_blur, name = 'gaussian-blur'),
    path('median-blur/', views.median_blur, name = 'median-blur'),
    path('sobel-edge-detection/', views.sobel_edge_detection, name = 'sobel-edge-detection'),
    path('roberts-edge-detection/', views.roberts_edge_detection, name = 'roberts-edge-detection'),
    path('prewitt-edge-detection/', views.prewitt_edge_detection, name = 'prewitt-edge-detection'),
    path('canny-edge-detection/', views.canny_edge_detection, name = 'canny-edge-detection'),
    path('get-histogram/', views.get_histogram, name = 'get-histogram'),
    path('get-equalized-histogram/', views.get_equalized_histogram, name = 'get-equalized-histogram'),
    path('get-equalized-image/', views.get_equalized_image, name = 'get-equalized-image'),
    path('normalize/', views.normalize, name = 'normalize'),
    path('global-threshold/', views.global_threshold, name = 'global-threshold'),
    path('local-threshold/', views.local_threshold, name = 'local-threshold'),
    path('get-hybrid-image/', views.get_hybrid_image, name = 'get-hybrid-image'),
]
