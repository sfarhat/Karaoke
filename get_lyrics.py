import requests
import bs4

def find_lyrics(song_name, artist_name):

    payload = {"q": song_name + " " + artist_name + " lyrics"}
    url = "https://google.com/search"

    request_result = requests.get(url, params=payload)
    soup = bs4.BeautifulSoup(request_result.text, "html.parser")

    lyrics = soup.find_all(class_="hwc")[0].text

    f = open("test.txt", "w")
    f.write(lyrics)
    f.close()

    return lyrics

if __name__ == "__main__":
    song_name = input("What song would you like the lyrics for?")
    artist_name = input("Who is the artist?")
    print(find_lyrics(song_name, artist_name))