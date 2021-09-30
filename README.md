# pdf-to-speech
I have issues reading pdfs and listening to them helps me out massively! So I'm working on a user-friendly application that can be given a pdf (or txt file) and spit out a WAV or MP3 file.
In the future, this'll be a fun server that'll do the hard work, but for now, it'll just be a python/bash project.
This is a small personal project, so there won't be regular updates *per se*, but when I have time I'll push what I've got.

## Requirements
Currently only tested on linux (primarily fedora, partially on ubuntu). On debian/ubuntu, run
`sudo apt install -y python3 python3-venv espeak ffmpeg tesseract-ocr-all python3-dev libenchant-dev libpoppler-cpp-dev pkg-config libavcodec libavtools ghostscript poppler-utils`
and on any platform (preferably in a virtualenv):
`pip install ocrmypdf transformers TTS pydub nltk pyspellchecker atlastk`
And get pytorch
Takes ~2-3GB :/

## What works now
test.py can turn all pdfs in a folder called in/ into mp3s in a folder called out/, with the full ocr->BERT->tts pipeline (it is advisable to run fixswap.sh or otherwise get lots of memory for this as longer texts can get quite large).
On my current setup (4 intel i7 8th gen cores, no gpu, debian 10, 5gb ram+7gb swap) takes 0.128*(word count)-52.616 seconds (r^2=0.949,n=3), which is actually pretty good, clocking in at around 10 words per second with some overhead. Unfortunately, almost all of the pdfs I'm experimenting with are in the 10s of thousands of words, which clocks in at around half an hour, which is less good for getting through my backlog. Ah well.

## Automated Pipeline
When everything works, this'll probably be how it fits together:
input.pdf -> ocrmypdf (ghostscript->unpaper->tesseract-ocr) -> preprocessing (regex) -> ocr correction (BERT) -> postprocessing (regex) -> text to speech (Mozilla TTS) -> wav to mp3 (pydub~ffmpeg) -> out.mp3
(a non-python bash workflow can be found in `readaloud.sh`, which I was using before I started this project)
The slowest parts will almost certainly be BERT and TTS, so it'd be nice to train student models for those when I have resources.
Hopefully this can all be controlled by a nice simple ui.
Hopefully.
Also be packaged nicely.
