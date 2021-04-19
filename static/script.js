$(document).ready(() => {
    let songName, artistName, animateHighlight, lineByLine;
    $("#audio").hide();


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

        let transcript = alignJSON.transcript;
        // This is an important edge case, makes handling first line not any different
        transcript = "\r\n" + transcript;
        let words = alignJSON.words;
        // console.log(JSON.stringify(transcript));

        let currOffset = 0;
        words.forEach((word, i) => {
            let elem;
            word.break = false;

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

            // Problematic edge cases have to do with start and end timings
            // <unk> words don't pose problems as they have well-defined start and end times
            // Note: weird edge case where successful word has 0 length (start === end), no fix for now
            if (word.case === "not-found-in-audio")  {
                // Handles edge case of "unfound" words having undefined start and end times
                // Will take end time of previous "found" word and start time of next "success" word
                // Divide this period of time equally among all "unfound" word between these

                // console.log(word)

                let numUnsuccessful = 1;
                let indexWithinUnsuccessful = 0;

                // Look to the right without hitting end of transcript
                let right = i + 1;
                while (right < words.length && words[right].case === "not-found-in-audio") {

                    numUnsuccessful++;
                    right++;
                }

                // Edge case for reaching past last word in transcript
                let nextStart = right === words.length ? $("#audio").duration : words[right].start;

                // Look to the left without hitting beginning of transcript
                let left = i - 1;
                while (left > -1 && words[left].case === "not-found-in-audio") {

                    numUnsuccessful++;
                    indexWithinUnsuccessful++;
                    left--;
                }

                // Edge case for reaching before first word in transcript
                let prevEnd = left === -1 ? 0 : words[left].end;

                let duration = (nextStart - prevEnd) / numUnsuccessful;
                word.start = prevEnd + (duration * indexWithinUnsuccessful);
                word.end = prevEnd + (duration * (indexWithinUnsuccessful + 1));
            } 

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

    function scrollToLine(words, highlightIndex) {

        let scrolledTo = words[highlightIndex];

        scrolledTo.elem.scrollIntoView({
            behavior: "smooth",
            block: "center",
            inline: "center"
        });


    }
    
    let currHighlightedWord;
    let highlightDuration;
    function highlightWord(words) {

        let t = $("#audio")[0].currentTime; // index of multiple sources
        // Get index for highlighting and position for to help w/ showing line
        let highlightIndex = words.findIndex(word => (word.start <= t && word.end >= t));
        let highlightedWord = words[highlightIndex];
    
        if (highlightIndex !== -1) {
            // Audio has started, so highlightedWord should be defined
            if (highlightedWord !== currHighlightedWord) {
                // If first word in line, will add whole line, else should do nothing
                if (highlightedWord.break) {
                    // reached new line
                    if(lineByLine) {
                        console.log("Showing line beginning with word: " + highlightedWord.word);
                        showLine(words, highlightIndex);
                    } else {
                        console.log("Scrolling to line beginning with word: " + highlightedWord.word);
                        scrollToLine(words, highlightIndex);
                    }
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
        // TODO: UI/UX: on submit, make form unusable so as to not mess with anything
        // check console to see what this means

        songName = $("#song-name").val();
        artistName = $("#artist-name").val();
        animateHighlight = $("#animate-highlight").is(":checked");
        lineByLine = $("#line-by-line").is(":checked");


        let params = new URLSearchParams();
        params.append("song-name", songName);
        params.append("artist-name", artistName);

        // Using GET
        $("#progress").append("<p>Request received, fetching lyrics...</p>");
        fetch("/lyrics?" + params.toString())
        .then(lyricsResponse => {
            if (!lyricsResponse.ok) {
                throw new Error("Lyric retrieval failed. Probably because Google thinks you're a robot. Please try again in ~10 minutes.");
            }
            $("#progress").append("<p>Lyrics retrieved, getting audio...</p>");
            return fetch("/audio?" + params.toString());
        })
        .then(audioResponse => {
            if (!audioResponse.ok) {
                throw new Error("Audio retrieval failed. Probably because Youtube doesn't have the audio.")
            }
            $("#progress").append("<p>Audio retrieved, separating it into vocals and accompaniament...</p>");
            return fetch("/source-separator");
        })
        .then(separationResponse => {
            if (!separationResponse.ok) {
                throw new Error("Source separation failed.")
            }
            $("#progress").append("<p>Audio separated, getting alignment...</p>")
            return fetch("/alignment");
        })
        .then(alignmentResponse => {
            if (!alignmentResponse.ok) {
                throw new Error("Alignment failed.")
            }
            $("#progress").append("<p>Finished!</p>")
            return alignmentResponse.json();
        })
        .then(alignmentBody => {
            console.log(alignmentBody);
            // song.mp3 will be loaded in by Flask
            $("#audio").attr("src", "/music/song.mp3")
            $("#audio").show();
            processAlignment(alignmentBody);
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });
    });
});
