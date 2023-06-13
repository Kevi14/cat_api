import os
import dotenv
dotenv.read_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
print(os.environ.get("DJANGO_SECRET_KEY"))
DJANGO_SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    None,
)

API_MOUNT_PATH = os.environ.get(
    "API_MOUNT_PATH",
    "v1",
)


API_LIST_DEFAULT_PAGE_SIZE = int(
    os.environ.get(
        "API_LIST_DEFAULT_PAGE_SIZE",
        25,
    )
)
API_LIST_MAX_PAGE_SIZE = int(
    os.environ.get(
        "API_LIST_MAX_PAGE_SIZE",
        50,
    )
)

DATABASES = {
    'default': {
        'ENGINE': os.environ.get("DB_ENGINE", ""),
        'NAME': os.environ.get("DB_NAME", ""),
        'USER': os.environ.get("DB_USER", ""),
        'PASSWORD': os.environ.get("DB_PASSWORD", ""),
        'HOST': os.environ.get("DB_HOST", ""),
        'PORT': os.environ.get("DB_PORT", ""),
    }
}