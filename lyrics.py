import requests
import bs4
from flask import Flask, render_template, request 

app = Flask(__name__)

@app.route("/lyrics", methods=["GET", "POST"])
def lyrics():

    if request.method == "POST":
        result = request.json
        song_name, artist_name = result["song-name"], result["artist-name"]

    if request.method == "GET":
        song_name, artist_name = request.args.get("song-name"), request.args.get("artist-name")

    payload = {"q": song_name + " " + artist_name + " lyrics"}
    url = "https://google.com/search"

    request_result = requests.get(url, params=payload)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

    lyrics = soup.find_all(class_="hwc")[0].text

    # return lyrics

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


@app.route("/")
def output():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)