from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from pdf-to-speech.text import TextProcessor
from pdf-to-speech.tts import Reader
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
reader = Reader("out/",lang='en')
print("> Done text preprocessing")
for text,name in zip(texts,files):
    reader.tts(text,name)
time_taken = time.time()-start_time
with open('time_data.csv','a') as f:
   writer = csv.writer(f)
   writer.writerow([wordcount,time_taken])
print(f"> Read {wordcount} words in {time_taken} seconds")
