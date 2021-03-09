set $out to /home/user/Documents/out
source /home/user/Documents/tts/bin/activate
for file in *.pdf; do ocrmypdf --force-ocr --rotate-pages --remove-background --deskew --clean --clean-final --sidecar $file.txt $file $file;done
for file in *.pdf; do pdftotext $file;done
for file in $name*.txt; do cat $file | tr '\n' ' ' | xargs -I % tts --text % --model_name "tts_models/en/ljspeech/tacotron2-DCA" --vocoder_name "vocoder_models/universal/libri-tts/fullband-melgan" --out_path $out/;done
for file in $out/*.wav; do echo "file '$file'">>$name.txt;done
ffmpeg -f concat -safe 0 $name.txt -c copy $name.wav
ffmpeg -i $name.wav $name.mp3
rm out/*
rm $name.wav
deactivate
