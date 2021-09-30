from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from text import TextProcessor
import numpy as np
import os
import time
import csv
import re

force_english = False

tag_remover = re.compile('<.*?>')

def get_ext(filename):
    return filename.split(".")[-1]

start_time = time.time()

os.environ["TOKENIZERS_PARALLELISM"] = "False"
tp = TextProcessor()
sesspath = "in/"
files = [f for f in os.listdir(sesspath) if get_ext(f) in ['pdf','txt','muse']]
texts = [[] for f in files]
wordcount = 0
print(f"> Reading {files}")
for i,filename in enumerate(files):
    if get_ext(filename) == 'pdf':
        text = tp.loadpdf(filename,sesspath,force=True,force_english=force_english)
    elif get_ext(filename) == 'txt':
        with open(sesspath+filename,'rt') as f:
            text = f.read()
        text = tp.correct_text(text,force_english=force_english)
    elif get_ext(filename) == 'muse':
        with open(sesspath+filename,'rt') as f:
            text = f.read()
        text = re.sub(tag_remover,'',text)
        text = tp.correct_text(text,force_english=force_english)
    else:
        continue
    wordcount += len(text.split(" "))
    texts[i] = text
del tp
print("> Done text preprocessing")
manager = ModelManager("venv/lib/python3.7/site-packages/TTS/.models.json")
model_path,config_path,_ = manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
vocoder_path,vocoder_config_path,_ = manager.download_model("vocoder_models/en/ljspeech/hifigan_v2")
synth = Synthesizer(model_path,config_path,vocoder_checkpoint=vocoder_path,vocoder_config=vocoder_config_path)
def tts(text,name,outpath):
    print(f"> Reading {name}")
    wav = synth.tts(text)
    wav = np.array(wav)
    wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav = wav.astype(np.int16)
    print(f"> Saving as {outpath}{name}.mp3")
    AudioSegment(
        wav.tobytes(), 
        frame_rate=synth.ap.sample_rate,
        sample_width=wav.dtype.itemsize, 
        channels=1
    ).export(outpath+name+'.mp3',format="mp3")
for text,name in zip(texts,files):
    tts(text,name,"out/")
time_taken = time.time()-start_time
with open('time_data.csv','a') as f:
   writer = csv.writer(f)
   writer.writerow([wordcount,time_taken])
print(f"> Read {wordcount} words in {time_taken} seconds")
