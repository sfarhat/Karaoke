$(document).ready(() => {
    let songName;
    const url = "3005_lyrics.txt"
    $("#submit").click(() => {
        songName = $("#song_name").val();
        console.log(songName);
        $("#lyrics").load(url)
    })
    
});