#!/bin/bash

sudo apt install ffmpeg
pip install -r requirements.txt

docker start aligner

export FLASK_APP=lyrics.py
export FLASK_ENV=development

flask run
