from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
import numpy as np
from pathlib import Path
from pdf_to_speech.util.text import split_into_sentences

models_dict = {
    'en': ['tts_models/en/ljspeech/tacotron2-DDC', 'vocoder_models/en/ljspeech/hifigan_v2'],
    'fr': ['tts_models/fr/mai/tacotron2-DDC', 'vocoder_models/universal/libri-tts/fullband-melgan'],
    'es': ['tts_models/es/mai/tacotron2-DDC', 'vocoder_models/universal/libri-tts/fullband-melgan'],
    'de': ['tts_models/de/thorsten/tacotron2-DCA', 'vocoder_models/de/thorsten/fullband-melgan'],
    'ja': ['tts_models/ja/kokoro/tacotron2-DDC', 'vocoder_models/ja/kokoro/hifigan_v1'],
    'nl': ['tts_models/nl/mai/tacotron2-DDC', 'vocoder_models/nl/mai/parallel-wavegan'],
    'zh': ['tts_models/zh-CN/baker/tacotron2-DDC-GST', None]
}

# later i'll figure out how to load TTS's .models.json

manager = ModelManager(Path(__file__).parent / "../.models.json")


class Reader:
    def __init__(self, outpath, lang='en', tts_name=None, voc_name=None):
        self.outpath = outpath
        model_name, vocoder_name = models_dict[lang]
        if tts_name is not None:
            model_name = tts_name
        if voc_name is not None:
            vocoder_name = voc_name
        model_path, config_path, _ = manager.download_model(model_name)
        if vocoder_name is None:
            self.synth = Synthesizer(model_path, config_path)
        else:
            vocoder_path, vocoder_config_path, _ = manager.download_model(voc_name)
            self.synth = Synthesizer(model_path, config_path, vocoder_checkpoint=vocoder_path,
                                     vocoder_config=vocoder_config_path)
        return

    def _decoder_split(self, text):
        raise NotImplementedError()

    def tts(self, text, fname):
        print(f"> Reading {fname}")
        max_len = max(len(s) for s in split_into_sentences(text))
        self.synth.tts_model.decoder.max_decoder_steps = max_len*3
        wav = self.synth.tts(text)
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
