from .base import *

DEBUG = False
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
DEFAULT_URL_SCHEME = "https"

db_host = os.environ.get("DB_HOST", default="db")
if os.environ.get("FHECLASSIFIER_PASSWORD"):
    fheclassifier_db_password = os.environ["FHECLASSIFIER_PASSWORD"]
elif SECRETS.get("DB_PASSWORD"):
    fheclassifier_db_password = SECRETS.get("DB_PASSWORD")
    db_host = SECRETS.get("DB_HOST")
else:
    with open(os.environ["FHECLASSIFIER_PASSWORD_FILE"]) as keyfile:
        fheclassifier_db_password = keyfile.read().strip()

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "fheclassifierdb",
        "USER": "fheclassifier",
        "PASSWORD": fheclassifier_db_password,
        "HOST": db_host,
        "PORT": "5432",
    }
}
