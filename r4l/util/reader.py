import os
from pathlib import Path

import nltk
import numpy as np
import psutil
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from tqdm import tqdm

from r4l import lang_dict


def split_into_sentences(string):
    try:
        sentences = nltk.sent_tokenize(string)
    except:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(string)
    return sentences


manager = ModelManager()


class Reader:
    def __init__(self, outpath, lang='en', tts_name=None, voc_name=None, decoder_mult=3, max_ram_percent=0.6):
        self.outpath = outpath
        self.decoder_mult = decoder_mult
        self.max_ram_percent = max_ram_percent
        model_name, vocoder_name, _ = lang_dict[lang]
        if tts_name is not None:
            model_name = tts_name
        if voc_name is not None:
            vocoder_name = voc_name
        print(model_name, vocoder_name)
        model_path, config_path, _ = manager.download_model(model_name)
        if vocoder_name is None:
            self.synth = Synthesizer(model_path, config_path)
        else:
            vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
            self.synth = Synthesizer(model_path, config_path, vocoder_checkpoint=vocoder_path,
                                     vocoder_config=vocoder_config_path)
        return

    def _write_to_file(self, wav, fname):
        wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav = wav.astype(np.int16)
        fout = self.outpath + fname + '.mp3'
        AudioSegment(
            wav.tobytes(),
            frame_rate=self.synth.ap.sample_rate,
            sample_width=wav.dtype.itemsize,
            channels=1
        ).export(fout, format="mp3")
        print(f"| > Wrote {fout}")
        return fout, len(wav)/self.synth.ap.sample_rate

    def tts(self, text, fname, manual_swap=True):
        print(f"> Reading {fname}")
        sens = split_into_sentences(text)  # overrides TTS's uh, underwhelming, sentence splitter
        sens = [s for s in sens if len(s.split(' ')) >= 2]  # remove empty sentences
        wav = None
        mem_tot = psutil.virtual_memory().total
        print(f"> Have {mem_tot / (1024 * 1024)}GB of memory total")
        audio_time = 0
        splits = 0
        for sen in tqdm(sens):
            self.synth.tts_model.decoder.max_decoder_steps = len(sen) * self.decoder_mult  # override decoder steps
            sen = " ".join([s for s in self.synth.split_into_sentences(sen) if
                            len(s.split(" ")) >= 2])  # TTS crashes on null sentences. this fixes that i think
            if wav is None:
                wav = np.array(self.synth.tts(sen))
            else:
                wav = np.append(wav, self.synth.tts(sen))
            if manual_swap:
                mem_tot = psutil.virtual_memory().total
                mem_use = psutil.Process().memory_info().rss
                print(f"> {100 * mem_use / mem_tot}% memory used")
                # Is the current RAM usage too high? write wav to file
                if mem_use / mem_tot > self.max_ram_percent:
                    self._write_to_file(wav, fname + str(splits))
                    splits += 1
                    wav = None
        audio_time = 0
        file = ""
        if wav is not None and splits > 0:
            self._write_to_file(wav, fname + str(splits))
            splits += 1
            wav = None
        if splits > 0:
            audio = AudioSegment.silent()
            print(f"> Collecting {splits} files to final mp3")
            for i in tqdm(range(splits)):
                file = self.outpath + fname + f'{i}.mp3'
                audio += AudioSegment.from_mp3(file)
                os.remove(file)
            audio_time = len(audio) / 1000
            audio.export(self.outpath + fname + '.mp3', format='mp3')
        elif wav is not None and splits == 0:
            file, audio_time = self._write_to_file(wav, fname)
        else:
            raise Exception("Somehow r4l.util.reader.wav is None")
        print(f"> Saved as {file}.mp3")
        return self.outpath + fname + '.mp3', audio_time
