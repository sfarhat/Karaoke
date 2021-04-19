# ubuntu:20.04, python:3.8 unecessarily large
# python:3.8-alpine makes pip run slow
FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y ffmpeg 

COPY requirements.txt /tmp/

WORKDIR /tmp

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR  /app

COPY static ./static
COPY templates ./templates 
COPY app.py . 

ENV FLASK_APP app.py
ENV FLASK_ENV development
ENV FLASK_RUN_PORT 5000
ENV FLASK_RUN_HOST 0.0.0.0

CMD ["flask", "run"]