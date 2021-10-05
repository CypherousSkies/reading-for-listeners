from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
import numpy as np
from pathlib import Path
from r4l import lang_dict

# later i'll figure out how to load TTS's .models.json

manager = ModelManager(Path(__file__).parent / "../.models.json")

class Reader:
    def __init__(self, outpath, lang='en', tts_name=None, voc_name=None):
        self.outpath = outpath
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

    def _decoder_split(self, text):
        raise NotImplementedError()

    def tts(self, text, fname):
        print(f"> Reading {fname}")
        sens = [s for s in self.synth.split_into_sentences(text) if len(s.split(' '))>=2]
        print(len(sens))
        max_len = max([len(s) for s in sens])
        self.synth.tts_model.decoder.max_decoder_steps = max_len*3
        wav = self.synth.tts(' '.join(sens))
        wav = np.array(wav)
        wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        wav = wav.astype(np.int16)
        print(f"> Saving as {self.outpath}{fname}.mp3")
        AudioSegment(
            wav.tobytes(),
            frame_rate=self.synth.ap.sample_rate,
            sample_width=wav.dtype.itemsize,
            channels=1
        ).export(self.outpath + fname + '.mp3', format="mp3")
        return self.outpath + fname + '.mp3'
