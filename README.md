# Reading for Listeners (r4l)
I have issues reading pdfs and listening to them helps me out massively! So I'm working on a user-friendly application that can be given a pdf (or txt file) and spit out an MP3 file.
In the future, this'll be a fun server that'll do the hard work, but for now, it'll just be a python/bash project.
This is a small personal project, so there won't be regular updates *per se*, but when I have time I'll push what I've got.

## Features
### Holistic OCR Improvement
The biggest problem with PDFs is they either don't have text within the document (are essentially images) or the existing text (usually the result of OCR) is of poor quality. 
OCR is often pretty bad on pdfs that I am given, so I use BERT (a masked language model) to improve spell-check results. In future this'll be replaced by Microsoft's TrOCR.
### TTS with Inflection
If OCR was the only problem, I'd just use make ocrmypdf output to espeak and we'd be done. Unfortunately, espeak sounds terrible. There's no inflection and it's *really* hard to pay attention to it for long periods of time.
That's where [Coqui.ai's TTS](https://github.com/coqui-ai/TTS) comes to the rescue, making hours-long readings bearable.
### Always FOSS
The other solutions to this problem are closed source and cost a *lot* of money. This is free.
### Simple UI
Eventually this project will have a neat web UI which'll require very little input from the end user.
This is accessibility software after all -- it would be weird if it was hard to use.
Unfortunately, for now I only have a cli which is only been tested on linux. Not the best, but I gotta start somewhere.

## Install

### Windows (WIP)
The "easiest" way of doing this is by installing [WSL](https://docs.microsoft.com/en-us/windows/wsl/) with Ubuntu and follow the Ubuntu/debian instructions.

If you're fancy and know how to python on windows, tell me how it goes and how you did it!

Note: unfortunately, it's hard to set up gpu stuff for WSL, and even then only really works for CUDA (NVIDIA) cards, which I have no way of testing as of now (not that I could test any gpu stuff now, but that's beyond the point).

### Mac (WIP)
Gotta say, I have no idea how to get all the dependencies (see ubuntu/debian) on mac. A cursory glance says that `brew` or `port` should be able to get most of them, but I have no idea about their availability. If you have a mac and figured this out, let me know how you did it!

### Ubuntu/Debian (tested)
`sudo apt install -y python3 python3-venv espeak ffmpeg tesseract-ocr-all python3-dev libenchant-dev libpoppler-cpp-dev pkg-config libavcodec libavtools ghostscript poppler-utils`

Make and activate a [virtual environment](https://docs.python.org/3/tutorial/venv.html), get [pytorch](https://pytorch.org), then run

`pip install reading4listeners`

And you're all set to run `r4l` (see below for usage info)

### Install from source (debian)
On debian, run

`sudo apt install -y python3 python3-venv espeak ffmpeg tesseract-ocr-all python3-dev libenchant-dev libpoppler-cpp-dev pkg-config libavcodec libavtools ghostscript poppler-utils`

`git clone https://github.com/CypherousSkies/pdf-to-speech`

`cd pdf-to-speech`

`python3 -m venv venv`

`souce venv/bin/activate`

`pip install -U pip setuptools wheel cython`

get [pytorch](https://pytorch.org)

`python setup.py develop`

Takes ~2-3GB of disk space for install

## Usage
`r4l [--in_path in/] [--out_path out/] [--lang en]` runs the suite of scanning and correction on all compatible files in the directory `in/` and  outputs mp3 files to `out/` using the language `en` (square brackets denoting optional parameters with default values).

Run `r4l --list_langs` to list supported languages

~~This program uses a lot of memory so I'd advise expanding your swap size by ~10GB (for debian use `fixswap.sh`)~~ (This should be fixed now, but if it runs out of memory/crashes randomly, increase swap size)

### Benchmarks
On my current setup (4 intel i7 8th gen cores, no gpu, debian 10, 5gb ram+7gb swap), the english setting will read around 440 words/min (N=21,R^2=0.97) into a 175 word/min audio file.
So r4l takes ~11.4 minutes to read a 5000 word file, which will take ~28.5 mins to listen to irl.

Unfortunately, I can't speed it up much beyond this for CPU-only systems.
The main sticking point was that file access is slow, but with improved RAM-awareness, the major slow-down is BERT and TTS, which are both designed to run quickly on G/TPU machines.

## Under the Hood
At a high level, here's how this works:

input.pdf -> ocrmypdf (ghostscript -> unpaper -> tesseract-ocr) -> preprocessing (regex) -> ocr correction (BERT) -> postprocessing (regex) -> text to speech (Coqui.ai TTS) -> wav to mp3 (pydub) -> out.mp3

### Future work
I'll almost certainly need to fine-tune TrOCR/BERT and TTS to better deal with the texts I'm interested in when I get access to a ML rig, but until then, I'll keep using the off-the-shelf models.
Hopefully this can all be controlled by a nice, simple web ui and left running on a server for public use.
Also I'd like to package this into an executable which requires minimal technical knowledge to use and maintain, but that's a far-off goal.
