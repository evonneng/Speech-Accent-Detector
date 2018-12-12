#create directory to hold all wav files
if [ ! -d "recordings_wav" ]; then
	mkdir "recordings_wav";
fi

#convert mp3 to wav
for file in $(ls recordings); do
    if [ ${file: -4} == ".mp3" ]
    then
    	#parse file for language name
    	dir_name="${file%'.mp3'}"
    	dir_name="${dir_name//[[:digit:]]/}"

    	#create directory for language
    	if [ ! -d "./recordings_wav/${dir_name}" ]; then
			mkdir ./recordings_wav/${dir_name};
		fi

		#convert and store wav in recordings_wav/name_of_language/
        if [ ! -e "./recordings_wav/${dir_name}/${file%.*}.wav" ]; then
        	ffmpeg -i recordings/$file ./recordings_wav/${dir_name}/${file%.*}.wav;
        fi
    fi
done

