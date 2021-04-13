import requests
import bs4
from flask import Flask, render_template, request 
import youtube_dl

app = Flask(__name__)


def get_lyrics(song_name, artist_name):

    # Extracting lyrics from Google
    payload = {"q": f"{song_name} {artist_name} lyrics"}
    url = "https://google.com/search"

    request_result = requests.get(url, params=payload)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

    lyrics = soup.find_all(class_="hwc")[0].text

    return lyrics

def get_alignment(lyrics):

    # Passing lyrics and audio into forced aligner
    # TODO: can move this logic to JS (any benefit?) once audio logic figured outt
    with open("lyrics.txt", "w") as f:
        f.write(lyrics)

    params = {'async': 'false'}

    files = {
        'audio': open('separated_vocals.wav', 'rb'),
        'transcript': open('old-lyrics.txt', 'rb')
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

    lyrics = get_lyrics(song_name, artist_name)

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
        search_query = song_name + " " + artist_name + " audio" 
        video = ydl.extract_info(f"ytsearch:{search_query}", download=True)['entries'][0]
        # video_url = video['webpage_url']
        # ydl.download([video_url])
        audio_file_name = f"{video['title']}-{video['id']}.{ydl_opts['postprocessors'][0]['preferredcodec']}"

    alignment_json = get_alignment(lyrics)
    return alignment_json

@app.route("/")
def output():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)