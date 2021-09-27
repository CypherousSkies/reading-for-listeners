from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
from text import TextProcessor
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
tp = TextProcessor()
sesspath = "in/"
files = [f for f in os.listdir(sesspath) if f[-3:]=='pdf']
texts = [[] for f in files]
print(f"> Reading {files}")
for i,filename in enumerate(files):
    text = [tp.loadtext(filename,sesspath,force=a) for a in [True,False]]
    text = [t for t in text if len(t)>100]
    if len(text) == 1:
        text = text[0]
    texts[i] = text
del tp
print(f"> Done text preprocessing")
manager = ModelManager(".models.json")
model_path,config_path,_ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
vocoder_path,vocoder_config_path,_ = manager.download_model("vocoder_models/universal/libri-tts/fullband-melgan")
synth = Synthesizer(model_path,config_path,vocoder_checkpoint=vocoder_path,vocoder_config=vocoder_config_path)
def tts(text,name,outpath):
    print(f"> Reading {name}")
    wav = synth.tts(text)
    #synth.save_wav(wav,name+'.wav')
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
    if isinstance(text,list):
        for t,f in zip(text,['force.','redo.']):
            tts(t,f+name,"out/")
    else:
        tts(text,name,"out/")
print("> Done batch")
