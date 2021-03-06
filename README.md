Karaoke 🎤
=====

![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/seanfarhat/karaoke) ![Docker Image Size (tag)](https://img.shields.io/docker/image-size/seanfarhat/karaoke/latest)

A web app that, given a song name and its artist, will display the full karaoke experience to sing along too. Complete with the song, lyrics, and a synchronized lyric guide.


![Showcase of Website](./showcase.gif)

Design
----

Karaoke has 3 main components:

1. Music to hear
2. Lyrics to read
3. A lyric guide in time with the audio

While the first 2 items can be straightforwardly scraped from the Internet with no prep necessary, the lyric guide (also known as the **Forced Alignment**) requires more heavy lifting. Specifically, it requires the audio and lyrics beforehand, and for a performance increase, the audio should be source separated to just be the vocals.

This directly motivates our design.

### Backend Design (Flask/Python)

- To get the song audio, I used the [`youtube-dl`](https://github.com/ytdl-org/youtube-dl) library to scrape Youtube for the video containing the song's audio, and extracted it via `ffmpeg`.
- To get the lyrics, I used the `requests` and `beautifulsoup4` libraries to scrape Google Search results, since they conveniently serve the raw lyrics.
- For improved alignment, I passed the song into a source separator. I chose to use [`Spleeter by Deezer`](https://github.com/deezer/spleeter) since they provided an easy to use Python API.
- To get the alignment, I passed in the lyrics and separated vocals into a forced aligner. I chose to use the [`Gentle Aligner`](https://github.com/lowerquality/gentle) in the form of a Docker container which functions as a REST API. Unfortunately, the official container on Docker Hub is not actively maintained, so I'm using this updated one instead [`cnbeining/gentle`](https://hub.docker.com/r/cnbeining/gentle)

### Frontend Design

1. An HTML form will receive the desired song/artist name and desired mode (for now karaoke is only supported)
2. Javascript uses the Fetch API to query the Flask backend with one POST request at the `/create-alignment` endpoint, creating and storing the relevant files in a directory named after the title of the Youtube video the song was downloaded from (this is for consistency among multiple requests mapping to the same song). For example, say we requested a song which downloaded its audio from a Youtube video called "Example Song". The directory would be organized as follows:

        - Example Song/
            - lyrics.txt
            - song.wav (used for audio player)
            - song/
                - vocals.wav (used for improved alignment)
                - accompaniment.wav (used for audio player if desired)
            - align.json

3. Display the fetched lyrics
4. Fetch the song from the `/audio/Example Song` endpoint and feed it into a player 
5. Fetch the alignment from the `/alignment/Example Song` endpoint and feed it into an animation pipeline that will highlight the appropriate word(s) in time with the song playing

This entire process takes between 1-2 minutes. The very first run will take a bit longer because it needs to download the pretrained models necessary for the aligner to work.

Deployment
----

To keep up to date with modern app/service development practices, I decided to container-ize this with Docker. The details of the built image can be found in [`Dockerfile`](https://github.com/sfarhat/Karaoke/blob/main/Dockerfile). By itself, the container will do everything except forced alignment of lyrics, which is done by another Docker container. So to make the overall architecture work, we need to Compose these 2 Docker images into one project. The implementation of this can be found in [`docker-compose.yml`](https://github.com/sfarhat/Karaoke/blob/main/docker-compose.yml).

### CI Pipeline
Every push to the `main` branch triggers an image build via Github Actions to [`seanfarhat/karaoke:latest`](https://hub.docker.com/r/seanfarhat/karaoke).

Installation
----

1. Download Docker
2. Clone this repository
3. `cd` into the directory where `docker-compose.yml` is
4. Run the following code in your terminal

        docker compose up
        
5. The app is now running on [`localhost:5000`](http://localhost:5000)

Alternatively, you can find the Docker image on [Dockerhub](https://hub.docker.com/r/seanfarhat/karaoke).

TODO
-----

Improve the UI
