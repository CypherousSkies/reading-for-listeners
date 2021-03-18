from text import TextProcessor
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment
import os
from pathlib import Path
import json
import justpy as jp

path = Path().cwd() / "TTS/TTS/.models.json"
manager = ModelManager(path)
print("> TTS Module Manager Loaded")
with open(path,"rt") as f:
    modelsdict = json.load(f)
tts_models = [f"tts_models/{lang}/{dataset}" for lang in modelsdict["tts_models"] for dataset in modelsdict["tts_models"][lang]]
vocoder_models = [f"vocoder_models/{lang}/{dataset}" for lang in modelsdict["vocoder_models"] for dataset in modelsdict["vocoder_models"][lang]]

tp = TextProcessor()
print("> Text Processor Loaded")

def initsynthesizer(model_name,vocoder_name,use_cuda):
    model_path, config_path = manager.download_model(model_name)
    vocoder_path, vocoder_config_path = manager.download(vocoder_name)
    return Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, use_cuda)

def tts(synth, text, out_name):
    wav = synth.tts(text)
    syth.save_wav(wav,out_name+'.wav')
    AudioSegment.from_wav(out_name+'.wav').export(out_name+'.mp3',format="mp3")
    os.remove(out_name+'.wav')

session_data = {}


title_style = "flex-auto text-xl font-semibold items-center"
subtitle_style = "w-full flex-none text-sm font-medium items-center"

title = jp.H1(text = "PDF to Speech", classes=title_style)
subtitle = jp.P(text = "A deep learning powered pipeline", classes=subtitle_style)
input_class = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
p_class = 'm-2 p-2 h-32 text-xl border-2'

async def main_screen():
    main = jp.WebPage(websockets=False)
    main += title
    main += subtitle
    def pdf_input(self,msg):
        print("event")
        sesspath = msg.session_id
        print(msg.session_id)
        if not os.path.isdir(sesspath):
            os.mkdir(sesspath)
        print(f"path made for {msg.session_id}")
        for i,v in enumerate(msg.form_data.files):
            print("getting file "+str(i))
            with open(f'{sesspath}/{v.name}','wb') as f:
                f.write(base64.b64decode(v.file_content))
        file_list = os.listdir(sesspath)
        print(f"> got {file_list}")
        if file_list:
            session_data[msg.session_id] = file_list
            msg.page.redirect = 'text_confirm'
            print(f"> got {file_list}")
    jp.P(text="Upload a pdf to get started",a=main,classes='w-full flex-none text-sm')
    form = jp.Form(a=main,enctype='multipart/form-data',submit=pdf_input)
    jp.Input(type='file',classes=jp.Styles.input_classes,a=form,accept='application/pdf',multiple=True)
    jp.Button(type='submit',text='Upload',classes=jp.Styles.button_simple,a=form)
    return main

@jp.SetRoute('/text_confirm')
async def text_confirm(request):
    sid = request.session_id
    sesspath = "/static/"+sid
    file_list = session_data[sid]
    texts = [tf.loadtext(f'{sesspath}/{f}',sesspath) for f in file_list]
    tc = jp.WebPage()
    tc += title
    jp.P(text="For each file, pick a version to turn into speech", classes=subtitle_style)
    form = jp.Form(a=tc,classes='border m-1 p-1 w-64')
    labels = [[] for txt in texts]
    for textz,f,ls in zip(texts,file_list,labels):
        ts = jp.Div(text=f,a=form,classes="grid grid-flow-col auto-cols-max md:auto-cols-min")
        for txt in textz:
            if txt is not None:
                l = jp.Label(a=ts,classes=input_class)
                jp.P(text=txt,a=l,classes=p_class)
                jp.Input(a=l,type='radio')
                ls += l
    confirm_button = jp.Input(value='Confirm', type='submit',a=form,classes='border m-2 p-2')
    def confirm(self,msg):
        print(msg)
    form.on('submit',confirm)



jp.justpy(main_screen)
