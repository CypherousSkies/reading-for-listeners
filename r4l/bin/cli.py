#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys

from r4l.util.text import TextProcessor
from r4l.util.reader import Reader, models_dict
import os
import time
import csv
import re

os.environ["TOKENIZERS_PARALLELISM"] = "False"
tag_remover = re.compile('<.*?>')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def get_ext(filename):
    return filename.split(".")[-1]


def get_texts(sesspath, lang, force_english):
    wordcount = 0
    tp = TextProcessor(sc_langs=lang)
    files = [f for f in os.listdir(sesspath) if get_ext(f) in ['pdf', 'txt', 'muse']]
    texts = [[] for _ in files]
    print(f"> Reading {files}")
    for i, filename in enumerate(files):
        if get_ext(filename) == 'pdf':
            text = tp.loadpdf(filename, sesspath, force=True, force_english=force_english)
        elif get_ext(filename) == 'txt':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            text = tp.correct_text(text, force_english=force_english)
        elif get_ext(filename) == 'muse':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            text = re.sub(tag_remover, '', text)
            text = tp.correct_text(text, force_english=force_english)
        else:
            continue
        wordcount += len(text.split(" "))
        texts[i] = text
    del tp
    print("> Done text preprocessing")
    return texts, files, wordcount


def read_texts(texts, files, outpath, lang):
    reader = Reader(outpath, lang=lang)
    for text, name in zip(texts, files):
        reader.tts(text, name)
    return


def main():
    parser = argparse.ArgumentParser(
        description="""Read PDFs into MP3 files!\n"""
                    """In the interests of user-friendliness, this cli will be kept pretty bare-bones"""
                    """
        Basic usage:
        $ r4l [--in_path in/] [--out_path out/] [--lang "en"]
        Converts pdfs, txts, muses in the folder "in/" and output mp3s to the folder "out/" with the primary language set to "en"
        List languages:
        $ r4l --list_languages
        Lists available languages (Warning! Not tested on non-latin scripts!)
            """
    )
    parser.add_argument("--in_path", type=str, default="in/", help="Path containing files to be converted.")
    parser.add_argument("--out_path", type=str, default="out/", help="Output path.")
    parser.add_argument("--lang", type=str, default="en", help="Two-letter language code.")
    parser.add_argument(
        "--list_langs",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="list available languages.",
    )
    args = parser.parse_args()
    if args.list_langs:
        print(models_dict.keys())
        sys.exit()
    if not os.path.isdir(args.in_path):
        print("input path must exist and contain files!")
        parser.parse_args(["-h"])
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    run(args.in_path, args.out_path, args.lang)
    return


def run(in_path, out_path, lang):
    start_time = time.time()
    force_english: bool = False
    texts, files, wordcount = get_texts(in_path, lang, force_english)
    read_texts(texts, files, out_path, lang)
    time_taken = time.time() - start_time
    with open('time_data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([wordcount, time_taken])
    print(f"> Read {wordcount} words in {time_taken} seconds")
    return


if __name__ == "__main__":
    run("in/", "out/", "en")
