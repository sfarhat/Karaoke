$(document).ready(() => {
    let songName;
    const lyricsUrl = "lyrics.txt";

    function processAlignment(alignJSON) {

        console.log(alignJSON);

        let transcript = alignJSON.transcript;
        let words = alignJSON.words;
        // $("#lyrics").html(lyrics);
        // console.log($("#lyrics").html());
        console.log(JSON.stringify(transcript));
        let currOffset = 0;
        words.forEach(word => {
            if (word.startOffset > currOffset) {
                // There is a newline or space
                let whitespace = transcript.slice(currOffset, word.startOffset);
                if (whitespace === "\r\n" || whitespace === "\r\n\r\n") {
                    $("#lyrics").append("<br>");
                } else {
                    $("#lyrics").append(whitespace);
                }
            }

            let printed = transcript.slice(word.startOffset, word.endOffset);
            $("#lyrics").append(`<span>${printed}</span>`);
            currOffset = word.endOffset;
        });
    };

    $("#submit").click(() => {
        songName = $("#song_name").val();
        console.log(songName);
        // $("#lyrics").load(lyricsUrl);
        $.getJSON("align.json", processAlignment); 
    });
    
    // Use $("#audio").currentTime and words["start"] and words["end"] for syncing highlighting
    $("#audio").on({
        "play": function () {
            console.log("Audio playing!");
        },
        "pause": function() {
            console.log("Audio paused!");
        } 
    });
});