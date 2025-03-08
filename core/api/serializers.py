from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework import serializers
from ..models import UserImage

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['username'] = user.username
        # ...

        return token


class UserImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserImage
        fields = '__all__'
    
    def validate(self, data):
        request = self.context['request']
        if not request.FILES:
            raise serializers.ValidationError("An image must be provided")
        return data