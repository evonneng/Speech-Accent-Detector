for file in $(ls); do
    if [ ${file: -4} == ".mp3" ]
    then
        ffmpeg -i $file ../recordings_wav/${file%.*}.wav
    fi
done
