from ocrmypdf import ocr
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import re
import nltk
from enchant.checker import SpellChecker
from difflib import SequenceMatcher
from transformers import AutoTokenizer, TFAutoModelForMaskedLM, pipeline
from pdftotext import PDF
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

nltk.download('popular')
print("> NLTK setup")

def _need_ocr(inpdf,minwords):
    with open(inpdf,"rb") as f:
        pdf = PDF(f)
    text = ' '.join(pdf)
    word_list = text.replace(',','').replace('\'','').replace('.','').lower().split()
    return True, text
    if len(word_list) > minwords:
        return False, text
    else:
        return True, ""
    
def _get_text(inpdf,sesspath,language,unpaper_args,minwords):
    force_ocr, prelim_text = _need_ocr(inpdf,minwords)
    ocr(inpdf,f"{sesspath}/tmp.pdf",sidecar=f"{sesspath}/tmp.txt",language=language,deskew=force_ocr,rotate_pages=force_ocr,remove_background=force_ocr,clean=force_ocr,unpaper_args=unpaper_args,redo_ocr=(not force_ocr),force_ocr=force_ocr)
    with open(f"{sesspath}/tmp.txt","rt") as text:
        return text.read(), prelim_text

# from https://stackoverflow.com/a/16826935
def _remove_citations(text):
    author = "(?:[A-Z][A-Za-z'`-]+)"
    etal = "(?:et al.?)"
    additional = "(?:,? (?:(?:and |& )?" + author + "|" + etal + "))"
    year_num = "(?:19|20)[0-9][0-9]"
    page_num = "(?:, p.? [0-9]+)?"  # Always optional
    year = "(?:, *"+year_num+page_num+"| *\("+year_num+page_num+"\))"
    regex = "(" + author + additional+"*" + year + ")"
    return re.sub(regex,'',text)

# from Ravi Ilango's Medium post
def _get_personslist(text):
    print("> removing names and places")
    personslist=[]
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0,(chunk.leaves()[0][0]))
    return list(set(personslist))

def _cleanup(text):
    rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', '(': ' ( ', ')': ' ) ', "s'": "s '", ":": " ", "[":' [ ', "]":' ] ',"{":" { ","}":" } "}
    rep = dict((re.escape(k),v) for k,v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))],text)

class TextProcessor(object):
    def __init__(self,sc_language="en_US",bert_model="distilbert-base-multilingual-cased"):
        self.sc = SpellChecker(sc_language)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = TFAutoModelForMaskedLM.from_pretrained(bert_model)
        print("> BERT loaded")

    # take input pdf (inpdf) plus a bunch of settings and output a the fixed output (of both the new ocr and existing text layer)
    def loadtext(self,inpdf,sesspath,minwords=10,minletter=2,ocr_lang=['eng'],unpaper_args=None,remove_citations=True):
        print(f"> loading {inpdf}")
        textocr, pretext = _get_text(inpdf,sesspath,ocr_lang,unpaper_args,minwords)
        print(textocr[150:200],"---",pretext[150:200])
        os.remove(f'{sesspath}/tmp.pdf')
        os.remove(f'{sesspath}/tmp.txt')
        #os.remove(inpdf)
        texts = [textocr] + [pretext] if pretext is not None else []
        print(f"> processing {len(texts)}")
        for text in texts:
            ft,ot,sw = self._preprocess(text,remove_citations,minletter)
            text = self._predict_words(ft,ot,sw)
        #if len(texts) == 1:
        #    return text[0]
        return text

    def _preprocess(self,text,remove_citations,minletter): 
        if remove_citations:
            text = _remove_citations(text)
        text = text.replace('...',';')
        text = text.replace('. . .',';')
        text = text.replace('®','')
        text = text.replace('“','\"')
        text = text.replace('”',"\"")
        text = re.sub(r'https?://\S+','',text)
        original_text = text#.copy()
        text = _cleanup(text)
        personslist = _get_personslist(text)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]
        words = text.split()
        incorrectwords = list(set([w for w in words if not self.sc.check(w) and w not in ignorewords and len(w) >= minletter]))
        print(incorrectwords)
        suggestedwords = []
        mask = self.tokenizer.mask_token
        suggestedwords = self._gen_suggested(text,incorrectwords)
        print(suggestedwords)
        for w in incorrectwords:
            text = text.replace(w, mask)
            original_text = original_text.replace(w, mask)
        return text, original_text, suggestedwords
    def _gen_suggested(self,text,ws):
        sw = []
        if ws == []:
            return sw
        suggested = self.sc.suggest(ws[0])
        for txt in text.split(ws[0])[:-1]:
            sw.append(suggested)
            for l in self._gen_suggested(txt,ws[1:]):
                if l is not []:
                    sw.append(l)
        for l in self._gen_suggested(text.split(ws[0])[-1],ws[1:]):
            if l is not []:
                sw.append(l)
        return sw
    #from https://towardsdatascience.com/tensorflow-and-transformers-df6fceaf57cc
    def _tokenize(text):
        sensplitter = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = senssplitter.tokenize(text.strip())
        print(f"> split {len(sentences)}")
        ids = np.zeros(len(sentences),self.seq_len)
        mask = np.zeros(len(sentences),self.seq_len)
        for i,sentence in enumerate(tqdm(sentences)):
            tokens = self.tokenizer.encode_plus(sentence,max_length=self.seq_len,
                    truncation=True,padding='max_length',
                    add_special_tokens=True,return_attention_mask=True,
                    return_token_type_ids=False,return_tensors='tf')
            ids[i,:] = tokens['input_ids']
            mask[i,:] = tokens['attention_mask']
        return ids,mask
    def _predict_words(self,text_filtered, text_original, suggestedwords):
        print(f"> predicting {len(suggestedwords)} words")
        # BERT time
        input_ids,mask = self._tokenize(text_filtered)
        predictions = self.model(input_ids,attention_mask=mask)[0]
        print(predictions)
        print("> done predicting")
        # refine BERT predictions with spellcheck
        for i,m in enumerate(maskids):
            preds = torch.topk(predictions[0][m],k=50)
            indices = preds.indices.tolist()
            list1 = self.tokenizer.convert_ids_to_tokens(indices)
            list2 = suggestedwords[i]
            #print(list1)
            #print(list2)
            simmax=0
            predicted_token=list1[0]
            for word1 in list1[1:]:
                for word2 in list2:
                    s = SequenceMatcher(None,word1,word2).ratio()
                    if s is not None and s > simmax:
                        simmax = s
                        predicted_token = word1
            text_original = text_original.replace(self.tokenizer.mask_token,predicted_token,1)
        print("> done processing text")
        return text_original
