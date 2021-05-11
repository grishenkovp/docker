FROM python:slim-buster

LABEL maintainer="Pavel"
LABEL version="1.0"
LABEL description="Приложение по классификации цветов"

WORKDIR /app 

COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

