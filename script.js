$(document).ready(() => {
    let songName;
    const lyricsUrl = "lyrics.txt";
    let audio = $("#audio");
    audio.hide();

    function processAlignment(alignJSON) {

        console.log(alignJSON);

        let transcript = alignJSON.transcript;
        let words = alignJSON.words;
        // console.log(JSON.stringify(transcript));

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

            // Want to access these elements from list in animation function
            let text = document.createTextNode(transcript.slice(word.startOffset, word.endOffset));
            let elem = document.createElement("span");
            elem.appendChild(text);
            $("#lyrics").append(elem);
            word.elem = elem;

            currOffset = word.endOffset;
            });
        
        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    let currWord;
    function highlightWord(words) {
        let t = audio[0].currentTime; // index becuase multiple sources
        // Should only be one word that fits within timestamps
        let shouldBeHighlighted = words.filter(word => (word.start <= t && word.end >= t))[0];
        if (shouldBeHighlighted && shouldBeHighlighted != currWord) {
            currWord = shouldBeHighlighted;
            console.log(currWord)
            shouldBeHighlighted.elem.classList.add("highlighted");
        }
        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    $("#submit").click(() => {
        songName = $("#song_name").val();
        console.log(songName);
        audio.show();
        // $("#lyrics").load(lyricsUrl);
        $.getJSON("align.json", processAlignment); 
    });
    
    // Use $("#audio").currentTime and words["start"] and words["end"] for syncing highlighting
    // $("#audio").on({
    //     "playing": function () {
    //         console.log("Audio playing!");
    //     },
    //     "pause": function() {
    //         console.log("Audio paused!");
    //     } 
    // });

    // while (audio.duration > 0 && !audio.paused)
});
