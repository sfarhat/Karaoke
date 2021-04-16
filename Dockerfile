FROM alpine:3.8

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    ffmpeg && \
    apt-get clean


COPY requirements.txt /app/

WORKDIR /app

RUN pip install -r requirements.txt

COPY . /app

ENV FLASK_APP lyrics.py
ENV FLASK_RUN_PORT 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
