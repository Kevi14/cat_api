from rest_framework_simplejwt.tokens import RefreshToken

from django.contrib.auth import get_user_model
from django.conf import settings
from rest_framework.exceptions import AuthenticationFailed

User = get_user_model()

def register_social_user(provider, user_id, email, name):
    filtered_user_by_email = User.objects.filter(email=email)

    if filtered_user_by_email.exists():
        registered_user = filtered_user_by_email[0]
        if provider == registered_user.auth_provider:
            registered_user.check_password(settings.SOCIAL_SECRET)

            refresh = RefreshToken.for_user(registered_user)
            token = str(refresh.access_token)
            refresh_token = str(refresh)

            return {
                # 'username': registered_user.username,
                'email': registered_user.email,
                'token': token,
                'refresh_token': refresh_token
            }
        else:
            raise AuthenticationFailed(
                detail="An account already exists with this email")
    else:

        user = User.objects.create_user(
            # username=email,
            first_name=name,
            last_name=name,
            email=email,
            password=settings.SOCIAL_SECRET,
            auth_provider=provider
        )
        user.is_active = True
        user.save()
        return {'email': email}
