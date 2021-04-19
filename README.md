# Karaoke 🎤

A web app that, given a song name and its artist, will display the full karaoke experience to sing along too. Complete with the song, lyrics, and a synchronized lyric guide.

### Backend Design

- To get the song audio, I used the [`youtube-dl`](https://github.com/ytdl-org/youtube-dl) library to scrape Youtube for the video containing the song's audio, and extracted it via ffmpeg.
- To get the lyrics, I used the `requests` and `beautifulsoup4` libraries to scrape Google Search results, since they conveniently serve the raw lyrics.
- For improved alignment, I passed the song into a source separator. I chose to use [`Spleeter by Deezer`](https://github.com/deezer/spleeter) since they provided an easy to use API.
- To get the alignment, I passed in separated vocals and lyrics into a forced aligner. I chose to use the [`Gentle Aligner`](https://github.com/lowerquality/gentle) in the form of a Docker container. Unfortunately, the official container on Docker Hub is not actively maintained, so I'm using this updated one instead [`cnbeining/gentle`](https://hub.docker.com/r/cnbeining/gentle)

These steps are handled by a Flask backend that will serve to the frontend the lryics-to-audio alignment JSON to parse, and the song audio to play.

### Frotnend Design

Using Javascript and jQuery, I GET the alignment from the backend, parse the resulting JSON, display the lyrics, and highlight the appropriate words in time with the timestamp of the audio playing

### TODO

Docker-ize the app to make deployment cleaner

Stylize the page for better UI/UX
