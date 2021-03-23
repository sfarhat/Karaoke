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
            let elem;
            word.break = false;
            if (word.startOffset > currOffset) {
                // There is a newline or space
                let whitespace = transcript.slice(currOffset, word.startOffset);
                if (whitespace === "\r\n" || whitespace === "\r\n\r\n") {
                    // To allow for showing each line, rather than entire lyrics,
                    // we add 'break' attribute to word after each <br>
                    elem = document.createElement("br");
                    word.break = true;
                    $("#lyrics").append(elem);
                } else {
                    $("#lyrics").append(whitespace);
                    // Don't worry about this for now since it messes with punctutaion.
                }
            }

            if (word.startOffset === 0) {
                // Edge case for very first word, add break for showLine to work properly
                word.break = true;
            }

            // Want to access these elements from list in animation function
            // Maybe slice until next space instead of endOffset to account for punctuation
            let text = document.createTextNode(transcript.slice(word.startOffset, word.endOffset));
            elem = document.createElement("span");
            elem.appendChild(text);
            elem.classList.add("hidden");
            $("#lyrics").append(elem);
            word.elem = elem;

            currOffset = word.endOffset;
            });
        
        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    let currShownWords = [];
    function showLine(words, highlightIndex) {
        // If first word in line, will add whole line, else should do nothing
        // Add back 'hidden' property from previously shown line
        currShownWords.forEach(word => {
            word.elem.classList.add("hidden");
        })

        // Clear shown words array to make room for new line
        currShownWords = [words[highlightIndex]];

        // Gather words until EOL 
        let right = highlightIndex + 1;
        while (!words[right].break) {
            currShownWords.push(words[right]);
            right++;
        }

        currShownWords.forEach(word => {
            word.elem.classList.remove("hidden");
        })
    }
    
    let currHighlightedWord;
    function highlightWord(words) {

        let t = audio[0].currentTime; // index becuase multiple sources
        // Get index for highlighting and position for to help w/ showing line
        let highlightIndex = words.findIndex(word => (word.start <= t && word.end >= t));
        let highlightedWord = words[highlightIndex];

        if (highlightIndex != -1) {
            // Audio has started, so highlightedWord should be defined
            if (highlightedWord != currHighlightedWord) {
                if (highlightedWord.break) {
                    // reached new line
                    console.log("Showing line beginning with word: " + highlightedWord.word);
                    showLine(words, highlightIndex);
                }
                currHighlightedWord = highlightedWord;
                console.log(currHighlightedWord);
                highlightedWord.elem.classList.add("highlighted");
            }
        }

        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    $("#submit").click(() => {
        songname = $("#song_name").val();
        console.log(songname);
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
