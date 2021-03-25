$(document).ready(() => {
    let songName, animateHighlight, lineByLine;
    const lyricsUrl = "lyrics.txt";
    let audio = $("#audio");
    audio.hide();


    function extractWord(transcript, word, currOffset) {
        // punctuation -> space/newline -> punctuation -> word -> punctuation -> space/newline
        // Example: asdf[? (a) ]asdf
        
        // Returns word including attached punctuation, 
        // while also adding first space/newline to DOM

        // This can probably be done with a regex expression but I don't wanna do that now

        // + 2 necessary to account for added "\r\n" at beginning
        // But for first word (offset = 0), +2 would skip over these characters, messing things up
        let i = currOffset === 0 ? currOffset : currOffset + 2;

        // First puncutation should be handled by call for previous word
        // So, move past it to reach first whitespace
        while (!(transcript[i] === " " || transcript[i] === "\n" || transcript[i] === "\r")) {
            i++
        }

        // Extract first whitespace
        let whitespace = "";
        while (transcript[i] === " " || transcript[i] === "\n" || transcript[i] === "\r") {
            whitespace += transcript[i];
            i++
        }

        // Process whitespace
        if (whitespace.includes("\n")) {
            // To allow for showing each line, rather than entire lyrics,
            // we add 'break' attribute to word after each newline character 
            // For nicer visuals, we don't add actual <br> to DOM but still use the 
            // conceptual "break" attribute to differentiate lines
            word.break = true;
            if (!lineByLine) {
                $("#lyrics").append(document.createElement("br"));
            }
        } 
       
        // Get rid of newlines and add space(s)
        $("#lyrics").append(whitespace.replace(/\r?\n|\r/g, ""));
    
        // Get puncutation + word + punctation 
        let text = ""; 
        while (i < transcript.length && 
            !(transcript[i] === " " || transcript[i] === "\n" || transcript[i] === "\r")) {
            text += transcript[i];
            i++;
        }

        return document.createTextNode(text);
    }

    function processAlignment(alignJSON) {

        console.log(alignJSON);

        let transcript = alignJSON.transcript;
        // This is an important edge case, makes handling first line not any different
        transcript = "\r\n" + transcript;
        let words = alignJSON.words;
        // console.log(JSON.stringify(transcript));

        let currOffset = 0;
        let prevEnd = 0;
        let nextStart = 0;
        words.forEach((word, i) => {
            let elem;
            word.break = false;

            // if (word.startOffset === 0) {
            //     // Edge case for very first word, add break for showLine to work properly
            //     // Alternate solution: could add "sentinel" empty line at begining of transcript
            //     word.break = true;
            // }

            // Want to access these elements from list in animation function
            let text = extractWord(transcript, word, currOffset);
            elem = document.createElement("span");
            elem.appendChild(text);
            if (lineByLine) {
                elem.classList.add("hidden");
            }
            elem.classList.add("lyric");
            $("#lyrics").append(elem);
            word.elem = elem;

            currOffset = word.endOffset;

            if (word.case === "not-found-in-audio") {
                // Handles edge case of "unfound" words not having start
                // or end times. This will just show the entire "unfound"
                // chunk for the unmatched times of the audio
                word.start = prevEnd;
                word.end = nextStart;
            }

            prevEnd = word.end;
            let j = i
            while (words[j].case === "not-found-in-audio") {
                j++;
            }
            nextStart = words[j].start;

            });
        
        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    let currShownWords = [];
    function showLine(words, highlightIndex) {
        // Hide previously shown line
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

        // Show words in line
        currShownWords.forEach(word => {
            word.elem.classList.remove("hidden");
        })
    }
    
    let currHighlightedWord;
    let highlightDuration;
    function highlightWord(words) {

        let t = audio[0].currentTime; // index becuase multiple sources
        // Get index for highlighting and position for to help w/ showing line
        let highlightIndex = words.findIndex(word => (word.start <= t && word.end >= t));
        let highlightedWord = words[highlightIndex];

        if (highlightIndex !== -1) {
            // Audio has started, so highlightedWord should be defined
            if (highlightedWord !== currHighlightedWord) {
                // If first word in line, will add whole line, else should do nothing
                if (highlightedWord.break && lineByLine) {
                    // reached new line
                    console.log("Showing line beginning with word: " + highlightedWord.word);
                    showLine(words, highlightIndex);
                }

                highlightedWord.elem.classList.add("highlight");

                highlightDuration = animateHighlight ? highlightedWord.end - highlightedWord.start : 0;
                highlightedWord.elem.style.transitionDuration = highlightDuration + "s";

                currHighlightedWord = highlightedWord;
                console.log(currHighlightedWord);
            }
        } 

        window.requestAnimationFrame(() => {
            highlightWord(words);
        });
    };

    $("#submit").click(() => {
        // TODO: add boolean check if song has already loaded to prevent duplicates
        // check console to see what this means
        songname = $("#song-name").val();
        animateHighlight = $("#animate-highlight").is(":checked");
        lineByLine = $("#line-by-line").is(":checked");
        console.log(songname);
        audio.show();
        // $("#lyrics").load(lyricsUrl);
        $.getJSON("align.json", processAlignment); 
    });
});
