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

    return lyrics

@app.route("/")
def output():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)