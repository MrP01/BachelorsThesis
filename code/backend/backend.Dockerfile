FROM python:3.8-alpine
WORKDIR /app

RUN apk update
RUN pip install --upgrade pip
RUN apk add --virtual build-deps gcc python3-dev musl-dev
RUN apk add postgresql-dev && pip install psycopg2
RUN apk del build-deps

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY . /app/
RUN chmod +x /app/manage.py /app/init.sh

VOLUME /var/resources/static/ /var/resources/media/
ENV DJANGO_SETTINGS_MODULE="FHEClassifier.settings.prod" \
    RESOURCE_DIR="/var/resources"
CMD crond -f -d 5 & gunicorn -b 0.0.0.0:8000 FHEClassifier.asgi
EXPOSE 8000
