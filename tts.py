# tts
ttsmodel = "tts_models/en/ljspeech/tacotron2-DDC"
vocmodel = "vocoder_models/en/ljspeech/hifigan_v2"
manager = ModelManager("venv/lib/python3.7/site-packages/TTS/.models.json")
model_path,config_path,_ = manager.download_model(ttsmodel)
vocoder_path,vocoder_config_path,_ = manager.download_model(vocmodel)
synth = Synthesizer(model_path,config_path,vocoder_checkpoint=vocoder_path,vocoder_config=vocoder_config_path)
# Ideally, find a way to predict how many decoder steps are going to be used, and shorten sentences to match
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
