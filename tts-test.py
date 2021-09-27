from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
import os
manager = ModelManager(".models.json")
model_path,config_path,_ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
vocoder_path,vocoder_config_path,_ = manager.download_model("vocoder_models/universal/libri-tts/fullband-melgan")
print(vocoder_path,vocoder_config_path)
synth = Synthesizer(model_path,config_path,vocoder_checkpoint=vocoder_path,vocoder_config=vocoder_config_path)
def tts(text,name,outpath):
    wav = synth.tts(text)
    synth.save_wav(wav,name+'.wav')
    del wav
    AudioSegment.from_wav(name+'.wav').export(outpath+name+'.mp3',format="mp3")
    os.remove(name+'.wav')
with open("in/txts/armedstruggle.txt","rt") as f:
    text = f.read()
wav = tts(text,"test","out/")
wav = np.array(wav)
wav = wav * (32767 / max(0.01, np.max(np.abs(wav))))
wav = wav.astype(np.int16)
AudioSegment(
    wav.tobytes(), 
    frame_rate=rate,
    sample_width=channel1.dtype.itemsize, 
    channels=1
).export("out/test.mp3",format="mp3")
