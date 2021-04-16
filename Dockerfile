FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y \
    python3.8 python3-dev python3-pip \
    ffmpeg && \
    apt-get clean

COPY requirements.txt /tmp/

WORKDIR /tmp

RUN pip3 install --no-cache-dir -r requirements.txt

WORKDIR  /app

COPY static ./static
COPY templates ./templates 
COPY lyrics.py . 

ENV FLASK_APP lyrics.py
ENV FLASK_ENV development
ENV FLASK_RUN_PORT 5000

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
