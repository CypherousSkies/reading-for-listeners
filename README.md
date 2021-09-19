# pdf-to-speech
I have issues reading pdfs and listening to them helps me out massively! So I'm working on a user-friendly application that can be given a pdf (or txt file) and spit out a WAV or MP3 file.
In the future, this'll be a fun server that'll do the hard work, but for now, it'll just be a python/bash project.
This is a small personal project, so there won't be regular updates *per se*, but when I have time I'll push what I've got.

## Requirements
Currently only tested on linux (fedora and ubuntu). On debian/ubuntu, run
`sudo apt install -y python3 python3-venv espeak ffmpeg tesseract-ocr-all python3-dev libenchant-dev libpoppler-cpp-dev pkg-config libavcodec libavtools ghostscript poppler-utils`
and on any platform (preferably in a virtualenv):
`pip install -U pip setuptools==57 wheel`
`pip install justpy`
`pip install -U setuptools`
`pip install ocrmypdf transformers TTS pydub nltk pyspellchecker`
And get pytorch

## What works now
readaloud.sh contains the barebones of a cli workflow, although the outputs of pdftotext and ocrmypdf are often mediocre at best, so this still requires a lot of person-time to edit so that Mozilla TTS doesn't freak out (if you get weird errors, be sure to remove elipses, double punctuation e.g. ?!, and special characters like @ and #).

## Automated Pipeline
When everything works, this'll probably be how it fits together:
input.pdf -> ocrmypdf (ghostscript->tesseract-ocr) -> preprocessing (regex) -> ocr correction (BERT) -> postprocessing (regex) -> text to speech (Mozilla TTS) -> wav to mp3 (pydub~ffmpeg) -> out.mp3
The slowest parts will almost certainly be BERT and TTS, so it'd be nice to train student models for those when I have resources.
Hopefully this can all be controlled by a nice simple ui.
Hopefully.
