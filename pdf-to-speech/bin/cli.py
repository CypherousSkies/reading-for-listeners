#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
from argparse import RawTextHelpFormatter
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from .util.text import TextProcessor
from .util.tts import Reader
import numpy as np
import os
import time
import csv
import re

os.environ["TOKENIZERS_PARALLELISM"] = "False"
tag_remover = re.compile('<.*?>')

def get_ext(filename):
    return filename.split(".")[-1]
def get_texts(sesspath):
    tp = TextProcessor()
    sesspath = "in/"
    files = [f for f in os.listdir(sesspath) if get_ext(f) in ['pdf','txt','muse']]
    texts = [[] for f in files]
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
    return texts
def read_texts(texts,outpath):
    reader = Reader(outpath,lang='en')
    for text,name in zip(texts,files):
        reader.tts(text,name)
    return

def main():
    parser = argparse.ArgumentParser(
            description="""Read PDFs into MP3 files!\n"""
            """In the interests of user-friendliness, this cli will be kept pretty bare-bones"""
            """
        Basic usage:
        $ ./pdf-to-speech/bin/cli.py [--in_path in/] [--out_path out/]
        will convert (english language) pdfs in the folder "in/" and output mp3s to the folder "out/"
            """
            )
    parser.add_argument("--in_path",type=str,default="in/",help="Path containing pdfs to be converted")
    parser.add_arguemnt("--out_path",type=str,default="out/",help="Output path")
    args = parser.parse_args()
    if not os.path.isdir(args.in_path):
        print("input path must exist and contain files!")
        parser.parse_args(["-h"])
    start_time = time.time()
    force_english = False
    wordcount = 0
    texts = get_texts(args.in_path)
    read_texts(texts,args.out_path)
    time_taken = time.time()-start_time
    with open('time_data.csv','a') as f:
       writer = csv.writer(f)
       writer.writerow([wordcount,time_taken])
    print(f"> Read {wordcount} words in {time_taken} seconds")
    return
if __name__ == "__main__":
    main()
