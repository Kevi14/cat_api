from django.contrib.auth.base_user import AbstractBaseUser, BaseUserManager
from django.db import models


class CustomUserManager(BaseUserManager):
        def create_user(self, email, first_name, last_name ,password=None,is_admin=False,auth_provider=""):
            """
            Creates and saves a User with the given email and password.
            """
            if not email:
                raise ValueError("Users must have an email address")

            user = self.model(
                email=self.normalize_email(email),
                first_name=first_name,
                last_name=last_name,
                is_admin=is_admin,
                auth_provider=auth_provider,
            )
            user.set_password(password)
            user.save(using=self._db)
            return user

        def create_superuser(self, email,first_name, last_name, password=None):
            user = self.create_user(
                email,
                first_name,
                last_name,
                password=password,
                is_admin=True,
            )
            return user




class User(AbstractBaseUser):
    CUSTOMER = "customer"
    PROVIDER = "provider"

    USER_TYPE_CHOICES = (
        (1, CUSTOMER),
        (2, PROVIDER),
    )
    is_admin = models.BooleanField(default=False)
    first_name = models.CharField("first name", max_length=150)
    last_name = models.CharField("last name", max_length=150)
    email = models.EmailField("email address", unique=True)
    auth_provider = models.CharField("auth provider", max_length=255, null=True, blank=True)
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ["first_name", "last_name"]
    @property
    def is_staff(self):
        return self.is_admin
    def __str__(self):
        return self.email

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True
    objects = CustomUserManager()



