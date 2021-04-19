import requests
import bs4
from flask import Flask, render_template, request, send_file 
import youtube_dl
from spleeter.separator import Separator
import os
import shutil

app = Flask(__name__)

@app.route("/lyrics")
def lyrics():

    print("BEGINNING LYRIC RETRIEVAL...")
    
    song_name, artist_name = request.args.get("song-name"), request.args.get("artist-name")

    # Extracting lyrics from Google
    payload = {"q": f"{song_name} {artist_name} lyrics"}
    url = "https://google.com/search"

    try:
        request_result = requests.get(url, params=payload)
        soup = bs4.BeautifulSoup(request_result.text, "html.parser")

        lyrics = soup.find_all(class_="hwc")[0].text
    except:
        cleanup_from_fail()
        return 'fail', 500

    # Saves lyrics to 'lyrics.txt'
    lyrics_file_name = 'lyrics.txt'
    with open(lyrics_file_name, "w") as f:
        f.write(lyrics)

    print("LYRIC RETRIEVAL COMPLETE")

    # Returns success with code 204 NO_CONTENT, empty response makes chrome think it failed
    return 'ok'

@app.route("/audio")
def audio():

    print("BEGINNING AUDIO RETRIEVAL...")

    song_name, artist_name = request.args.get("song-name"), request.args.get("artist-name")

    # Extracting audio from Youtube
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'cachedir': False
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            search_query = f"{song_name} {artist_name} audio" 
            video = ydl.extract_info(f"ytsearch:{search_query}", download=True)['entries'][0]
            # video_url = video['webpage_url']
            # ydl.download([video_url])
            audio_file_name = f"{video['title']}-{video['id']}.{ydl_opts['postprocessors'][0]['preferredcodec']}"
    except:
        cleanup_from_fail()
        return 'fail', 500

    # Save downloaded song to 'song.mp3' (assumes file downloaded is .mp3 codec which is specified in ydl_opts)
    os.rename(os.path.join(os.getcwd(), audio_file_name), 
                os.path.join(os.getcwd(), f"song.{ydl_opts['postprocessors'][0]['preferredcodec']}"))
            
    print("AUDIO RETRIEVAL COMPLETE")

    return 'ok'

@app.route("/source-separator")
def source_separator():

    # Must be called after /audio since it expects song.mp3 to exist
    # This creates 2 files: vocals/accompaniment.wav in folder named audio_file_name
    # TODO: consider using Docker container for this as well, can be called via Docker API
    print("SEPARATING AUDIO...")

    cwd = os.getcwd()
    audio_path = os.path.join(cwd, 'song.mp3')
    
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(audio_path, cwd)

    print("AUDIO SEPARATION COMPLETE")

    return 'ok'

@app.route("/alignment")
def alignment():

    # Must be fetched after /lyrics, /audio, and /source-separate since they put files where they should be
    print("GETTING ALIGNMENT...")

    cwd = os.getcwd()

    lyrics_path = os.path.join(cwd, 'lyrics.txt')

    # This naming convention is predetermined by Spleeter (./song name/vocals.wav)
    separated_audio_path = os.path.join(os.getcwd(), 'song', 'vocals.wav')

    # Passing lyrics and audio into forced aligner
    params = {'async': 'false'}

    files = {
        'audio': open(separated_audio_path, 'rb'),
        'transcript': open(lyrics_path, 'rb')
    }

    # payload = {'transcript': lyrics}

    # .text will just return text, whereas .json() should json-ify it but currently does nothing different...
    # Docker sets up a local network with each container reachable by http://{service-name}:{container port}
    try:
        alignment_json = requests.post('http://aligner:8765/transcriptions', params=params, files=files).json()
    except:
        cleanup_from_fail()
        return 'fail', 500

    print("ALIGNMENT RETRIEVED")

    # We have created lyrics.txt, song.mp3, and separated .wav files in the song directory
    # Only song.mp3 is necessary for the front-end, so we can delete the rest before returning
    os.remove(os.path.join(os.getcwd(), 'lyrics.txt'))
    shutil.rmtree(os.path.join(os.getcwd(), 'song'))

    return alignment_json

@app.route("/main", methods=["GET", "POST"])
def main():

    # This is useless now...
    if request.method == "POST":
        result = request.json
        song_name, artist_name = result["song-name"], result["artist-name"]

    if request.method == "GET":
        song_name, artist_name = request.args.get("song-name"), request.args.get("artist-name")

def cleanup_from_fail():

    # These are listed in creation order, so it will fail only after deleting everything that has already been created
    try:
        os.remove(os.path.join(os.getcwd(), 'lyrics.txt'))
        shutil.rmtree(os.path.join(os.getcwd(), 'song'))
        os.remove(os.path.join(os.getcwd(), 'song.mp3'))
    except OSError:
        pass

@app.route('/music/<path:filename>')
def download_file(filename):
    return send_file(filename)

@app.route("/")
def index():

    # on page reload, song.mp3 may be left over from previous run
    try:
        os.remove(os.path.join(os.getcwd(), 'song.mp3'))
    except OSError:
        pass

    return render_template("index.html")