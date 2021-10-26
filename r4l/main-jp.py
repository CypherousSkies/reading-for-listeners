import base64
import json
import os

import justpy as jp
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from pydub import AudioSegment

from r4l.util.text import TextProcessor

path = ".models.json"
manager = ModelManager(path)
print("> TTS Module Manager Loaded")
with open(path, "rt") as f:
    modelsdict = json.load(f)
tts_models = [f"tts_models/{lang}/{dataset}" for lang in modelsdict["tts_models"] for dataset in
              modelsdict["tts_models"][lang]]
vocoder_models = [f"vocoder_models/{lang}/{dataset}" for lang in modelsdict["vocoder_models"] for dataset in
                  modelsdict["vocoder_models"][lang]]

# put textprocessor into a queue
tp = TextProcessor()
print("> Text Processor Loaded")


def initsynthesizer(model_name, vocoder_name, use_cuda):
    model_path, config_path = manager.download_model(model_name)
    vocoder_path, vocoder_config_path = manager.download(vocoder_name)
    return Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, use_cuda)


def tts(synth, text, out_name):
    wav = synth.tts(text)
    syth.save_wav(wav, out_name + '.wav')
    AudioSegment.from_wav(out_name + '.wav').export(out_name + '.mp3', format="mp3")
    os.remove(out_name + '.wav')


session_data = {}

# https://justpy.io/tutorial/uploading_files/

title_style = "flex-auto text-xl font-semibold items-center"
subtitle_style = "w-full flex-none text-sm font-medium items-center"

title = jp.H1(text="PDF to Speech", classes=title_style)
subtitle = jp.P(text="A deep learning powered pipeline", classes=subtitle_style)
input_class = "m-2 bg-gray-200 border-2 border-gray-200 rounded w-64 py-2 px-4 text-gray-700 focus:outline-none focus:bg-white focus:border-purple-500"
p_class = 'm-2 p-2 h-32 text-xl border-2'


async def main_screen():
    main = jp.WebPage(websockets=False)
    main += title
    main += subtitle
    instruct = jp.P(text="Upload a pdf to get started", a=main, classes='w-full flex-none text-sm')

    def pdf_input(self, msg):
        sesspath = msg.session_id
        if not os.path.isdir(sesspath):
            os.mkdir(sesspath)
        for c in msg.form_data:
            if c.type == 'file':
                break
        print(f"path made for {msg.session_id}")
        instruct.text = "Working..."
        jp.P(a=main, text="Working...", classes=p_class)
        for i, v in enumerate(c.files):
            # bs = v.file_content.encode('utf-8')
            print("getting file " + v.name)
            print(type(v.file_content))
            with open(f'{sesspath}/{v.name}', 'wb') as f:
                # decoded = base64.decodebytes(bs)
                decoded = base64.b64decode(v.file_content, validate=True)
                if decoded[0:4] != b'%PDF':
                    raise ValueError('Missing the PDF file signature')
                print("done decoding")
                f.write(decoded)
                print(f"done writing {v.name}")
        file_list = os.listdir(sesspath)
        print(f"> got {file_list}")
        if file_list:
            texts = [[tp.loadtext(f'{sesspath}/{f}', sesspath), tp.loadtext(f'{sesspath}/{f}', sesspath, force=True)]
                     for f in file_list]
            session_data[msg.session_id] = (file_list, texts)
            msg.page.redirect = 'text_confirm'
            print(f"> got {file_list}")

    form = jp.Form(a=main, enctype='multipart/form-data', submit=pdf_input)
    jp.Input(type='file', classes=jp.Styles.input_classes, a=form, accept='application/pdf', multiple=True)
    jp.Button(type='submit', text='Upload', classes=jp.Styles.button_simple, a=form)
    return main


@jp.SetRoute('/text_confirm')
async def text_confirm(request):
    sid = request.session_id
    sesspath = sid + "/"
    (file_list, texts) = session_data[sid]
    tc = jp.WebPage()
    tc += title
    jp.P(text="For each file, pick a version to turn into speech", classes=subtitle_style)
    form = jp.Form(a=tc)  # ,classes='border m-1 p-1 w-64')
    labels = [[] for txt in texts]
    text_file = []
    for textz, f, ls in zip(texts, file_list, labels):
        ts = jp.Div(text=f, a=form, classes="grid grid-flow-col auto-cols-max md:auto-cols-min")
        for txt in textz:
            if txt is not None:
                l = jp.Label(a=form)
                l2 = jp.Label(text=txt, a=l)
                jp.Input(a=l, type='checkbox')
                ls += l
                text_file += [(txt, f)]
    confirm_button = jp.Input(value='Confirm', type='submit', a=form, classes='border m-2 p-2')

    def confirm(self, msg):
        to_speak = [text_file[c.id - 4] for c in msg.form_data if c.checked == True]
        session_data[sid] = to_speak
        print(to_speak)
        msg.page.redirect = 'choose_voice'
        print(msg)

    form.on('submit', confirm)
    return tc


@jp.SetRoute('/choose_voice')
async def choose_voice(request):
    sid = request.session_id
    sesspath = sid + "/"
    to_speak = session_data[sid]
    cv = jp.WebPage()
    cv += title
    jp.P(text="Pick TTS models.", classes=subtitle_style)
    form = jp.Form(a=cv)
    i = 2
    tts_ids = {}
    modelL = jp.Label(a=form)
    for model in tts_models:
        ml = jp.Label(text=model, a=modelL)
        jp.Input(a=ml, type='radio')
        i += 1
        tts_ids[i] = model
    voc_ids = []
    i += 1
    vocodL = jp.Label(a=form)
    for model in vocoder_models:
        vl = jp.Label(text=model, a=vocodL)
        jp.Input(a=vl, type='radio')
        voc_ids[i] = model

    def confirm(self, msg):
        print(msg)

    confirm_button = jp.Input(value='Confirm', type='submit', a=form, classes='border m-2 p-2')
    form.on('submit', confirm)


jp.justpy(main_screen)
