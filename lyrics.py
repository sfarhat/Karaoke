import requests
import bs4
from flask import Flask, render_template, request 
import youtube_dl
from spleeter.separator import Separator
import os

app = Flask(__name__)


def get_lyrics(song_name, artist_name):

    # Extracting lyrics from Google
    lyrics_file_name = 'lyrics.txt'
    payload = {"q": f"{song_name} {artist_name} lyrics"}
    url = "https://google.com/search"

    request_result = requests.get(url, params=payload)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

    lyrics = soup.find_all(class_="hwc")[0].text

    with open(lyrics_file_name, "w") as f:
        f.write(lyrics)

    return lyrics_file_name

def get_audio(song_name, artist_name):

    # Extracting audio from Youtube
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        search_query = f"{song_name} {artist_name} audio" 
        video = ydl.extract_info(f"ytsearch:{search_query}", download=True)['entries'][0]
        # video_url = video['webpage_url']
        # ydl.download([video_url])
        audio_file_name = f"{video['title']}-{video['id']}.{ydl_opts['postprocessors'][0]['preferredcodec']}"

    return audio_file_name

def get_alignment(lyrics_file_path, audio_file_path):

    # Passing lyrics and audio into forced aligner
    params = {'async': 'false'}

    files = {
        'audio': open(audio_file_path, 'rb'),
        'transcript': open(lyrics_file_path, 'rb')
    }

    # For now, store lyrics locally in file
    # TODO: look at local file storage for Docker?
    # payload = {'transcript': lyrics}

    # .text will just return text, whereas .json() should json-ify it but currently does nothing different...
    alignment_json = requests.post('http://localhost:8765/transcriptions', params=params, files=files).json()

    return alignment_json

@app.route("/lyrics", methods=["GET", "POST"])
def lyrics():

    if request.method == "POST":
        result = request.json
        song_name, artist_name = result["song-name"], result["artist-name"]

    if request.method == "GET":
        song_name, artist_name = request.args.get("song-name"), request.args.get("artist-name")

    cwd = os.getcwd()
    temp_dir = 'temp'
    # os.makedirs(temp_dir)

    # Download lyrics to file in cwd
    lyrics_file_name = get_lyrics(song_name, artist_name)
    # lyrics_path = os.path.join(cwd, temp_dir, lyrics_file_name)
    # Move to temp dir
    # os.rename(os.path.join(cwd, lyrics_file_name), lyrics_path)
    lyrics_path = os.path.join(cwd, lyrics_file_name)

    # Download audio to file in cwd
    audio_file_name = get_audio(song_name, artist_name)
    audio_path = os.path.join(cwd, audio_file_name)

    # Source separate audio and put result in temp dir
    separator = Separator('spleeter:2stems')
    # separator.separate_to_file(audio_path, os.path.join(cwd, temp_dir))
    separator.separate_to_file(audio_path, cwd)

    # audio path should be to separated vocals
    alignment_json = get_alignment(lyrics_path, audio_path)

    return alignment_json

@app.route("/")
def output():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)