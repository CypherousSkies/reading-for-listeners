#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import os
import re
import sys
import time
from datetime import timedelta
from reading4listeners import lang_dict
from reading4listeners.util.reader import Reader
from reading4listeners.util.text import TextProcessor

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


def get_texts(sesspath, lang):
    setup_time = time.time()
    tp = TextProcessor(langs=lang)
    setup_time = time.time()-setup_time
    files = [f for f in os.listdir(sesspath) if get_ext(f) in ['pdf', 'txt', 'muse']]
    run_times = {}
    word_counts = {}
    texts = [[] for _ in files]
    print(f"> Reading {files}")
    for i, filename in enumerate(files):
        print(f"> Loading {filename}")
        start = time.time()
        if get_ext(filename) == 'pdf':
            text = tp.loadpdf(filename, sesspath, force=True)
        elif get_ext(filename) == 'txt':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            text = tp.correct_text(text)
        elif get_ext(filename) == 'muse':
            with open(sesspath + filename, 'rt') as f:
                text = f.read()
            text = re.sub(tag_remover, '', text)
            text = tp.correct_text(text)
        else:
            continue
        run_times[filename] = time.time()-start+setup_time
        word_counts[filename] = len(text.split(" "))
        texts[i] = text
    del tp
    print("> Done text preprocessing")
    return texts, files, word_counts, run_times

def read_texts(texts, files, outpath, lang):
    setup_time = time.time()
    reader = Reader(outpath, lang=lang)
    setup_time = time.time()-setup_time
    run_times = {}
    audio_times = {}
    for text, name in zip(texts, files):
        start = time.time()
        _, t = reader.tts(text, name)
        run_times[name] = time.time()-start+setup_time
        audio_times[name] = t
    del reader
    return audio_times, run_times


def main():
    parser = argparse.ArgumentParser(
        description="""Read PDFs into MP3 files!\n"""
                    """In the interests of user-friendliness, this cli will be kept pretty bare-bones"""
                    """
        Basic usage:
        $ reading4listeners [--in_path in/] [--out_path out/] [--lang en]
        Converts pdfs, txts, muses in the folder "in/" and output mp3s to the folder "out/" with the primary language set to "en"
        List languages:
        $ reading4listeners --list_languages
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
    parser.add_argument(
        "--collect_time_data",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="write time taken data to `time_data.csv` for analysis",
    )
    args = parser.parse_args()
    if args.list_langs:
        print(list(lang_dict.keys()))
        sys.exit()
    if not os.path.isdir(args.in_path):
        print("input path must exist and contain files!")
        parser.parse_args(["-h"])
    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    run(args.in_path, args.out_path, args.lang, args.collect_time_data)
    return


def run(in_path, out_path, lang, time_data):
    start_time = time.time()
    texts, files, word_counts, run_t_times = get_texts(in_path, lang)
    audio_times, run_r_times = read_texts(texts, files, out_path, lang)
    time_taken = time.time() - start_time
    if time_data:
        with open('time_data.csv', 'a') as f:
            writer = csv.writer(f)
            for name in files:
                writer.writerow([get_ext(name),word_counts[name], run_t_times[name]+run_r_times[name], audio_times[name]])
    print(f"> Read {sum(word_counts.values())} words in {timedelta(seconds=time_taken)} seconds with a real time factor of {time_taken / sum(audio_times.values())}")
    return


if __name__ == "__main__":
    run("in/", "out/", "en", False)
