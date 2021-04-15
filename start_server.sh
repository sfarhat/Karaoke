#!/bin/bash

docker start aligner

export FLASK_APP=lyrics.py
export FLASK_ENV=development

flask run
