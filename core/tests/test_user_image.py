
from django.urls import reverse
from rest_framework import status
from rest_framework_simplejwt.tokens import AccessToken
from rest_framework.test import APITestCase
from django.contrib.auth.models import User
from core.models import UserImage


class UserImageTest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='password123',
        )
        self.user_image = UserImage.objects.create(
            user=self.user,
            image=None,
            out_image=None,
        )
