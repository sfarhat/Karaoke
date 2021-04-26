import requests
import bs4
from flask import Flask, render_template, request, send_file, send_from_directory
import youtube_dl
from spleeter.separator import Separator
import os
import json

app = Flask(__name__)

@app.route('/create-alignment', methods=['POST'])
def main():

    """
    This function will take the song/artist name and obtain all the necessary files for future alignment, and then align them
    It saves the files organized by the following directory structure:

    - <title from Youtube video that song is downloaded from>
        - lyrics.txt
        - song.wav
        - song
            - vocals.wav
            - accompaniment.wav
        - align.json

    It will return the name of the working directory that contains all relevant files for the request 
    """

    try:
        request_body = request.json
        song_name, artist_name = request_body['song-name'], request_body['artist-name']

        # 1. Pull audio from Youtube
        audio_file_name = 'song'
        audio_codec = 'wav'

        working_dir = get_audio(song_name, artist_name, audio_file_name, audio_codec)
        working_dir_path = os.path.join(os.getcwd(), working_dir)

        audio_path = os.path.join(working_dir_path, f"{audio_file_name}.{audio_codec}")
        if not os.path.exists(audio_path): 
            raise Exception("The audio failed to download for an unknown reason.")

        # 2. Pull lyrics from Google search
        lyrics_file_name = 'lyrics.txt'
        lyrics_path = os.path.join(working_dir_path, lyrics_file_name)

        # Path will exist if we have cached work from a previous request
        if not os.path.exists(lyrics_path):
            get_lyrics(song_name, artist_name, lyrics_path)

        # 3. Separate audio into vocals for alignment and accompaniment for audio player
        vocals_path = os.path.join(working_dir_path, audio_file_name, 'vocals.wav')
        accompaniment_path = os.path.join(working_dir_path, audio_file_name, 'accompaniment.wav')

        if not os.path.exists(vocals_path):
            source_separator(audio_path, working_dir_path)

        # 4. Get alignment
        alignment_file_name = 'align.json'
        alignment_path = os.path.join(working_dir_path, alignment_file_name)
        
        if not os.path.exists(alignment_path):
            get_alignment(alignment_path, lyrics_path, vocals_path)

    except Exception as e:
        return str(e), 500

    return working_dir    

def get_audio(song_name, artist_name, audio_file_name, audio_codec):

    # Extracting audio from Youtube
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_codec,
            'preferredquality': '192',
        }],
        'cachedir': False
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        search_query = f"{song_name} {artist_name} audio" 
        video = ydl.extract_info(f"ytsearch:{search_query}", download=True)['entries'][0]
        # video_url = video['webpage_url']
        # ydl.download([video_url])
        downloaded_file_name = f"{video['title']}-{video['id']}.{audio_codec}"

    # Make working dir with name video['title'] for all work for this request to be done in
    working_dir = video['title']
    working_dir_path = os.path.join(os.getcwd(), working_dir)

    os.makedirs(working_dir_path, exist_ok=True)

    os.rename(os.path.join(os.getcwd(), downloaded_file_name), 
                os.path.join(working_dir_path, f"{audio_file_name}.{audio_codec}"))

    return working_dir
            
def get_lyrics(song_name, artist_name, lyrics_path):

    # Extracting lyrics from Google
    payload = {"q": f"{song_name} {artist_name} lyrics"}
    url = "https://google.com/search"

    request_result = requests.get(url, params=payload)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

    # This assert will fail when Google throws a Captcha
    assert len(soup.find_all(class_='hwc')) > 0, 'Google thinks you are a bot. Please try again after a few minutes'

    lyrics = soup.find_all(class_="hwc")[0].text

    with open(lyrics_path, "w") as f:
        f.write(lyrics)

def source_separator(audio_path, working_dir_path):

    # This creates 2 files: vocals/accompaniment.wav in folder named <song-name> (which we have set to just be 'song') 
    # Should be done outside working_dir because pretrained_models should only be downloaded once for all requests
    # TODO: consider using Docker container for this as well, can be called via Docker API

    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, working_dir_path)

def get_alignment(alignment_path, lyrics_path, vocals_path):

    # Passing lyrics and audio into forced aligner
    params = {'async': 'false'}

    files = {
        'audio': open(vocals_path, 'rb'),
        'transcript': open(lyrics_path, 'rb')
    }

    # .text will just return text, whereas .json() should json-ify it but currently does nothing different...
    # Docker sets up a local network with each container reachable by http://{service-name}:{container port}
    try:
        alignment_json = requests.post('http://aligner:8765/transcriptions', params=params, files=files).json()
    except requests.exceptions.RequestException as e:
        raise Exception("Could not connect to aligner module. Is the appropriate Docker container running?")

    with open(alignment_path, "w") as f:
        json.dump(alignment_json, f)

@app.route('/audio/<path:dir_name>')
def get_audio_file(dir_name):
    return send_from_directory(dir_name, 'song.wav')

@app.route('/alignment/<path:dir_name>')
def get_alignment_file(dir_name):
    return send_from_directory(dir_name, 'align.json')

@app.route("/")
def index():
    return render_template("index.html")