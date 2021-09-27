import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
from nltk.tag import pos_tag
from spellchecker import SpellChecker
from difflib import SequenceMatcher
from ocrmypdf import ocr
import numpy as np
import os

def string_metric(str1,str2):
    score = SequenceMatcher(None,str1,str2).ratio()
    return score

class TextProcessor:
    def __init__(self,bert_model="distilbert-base-multilingual-cased",sc_langs=["en","fr"]):
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModelForMaskedLM.from_pretrained(bert_model)
        self.sc = SpellChecker(distance=1,language=sc_langs)
        print("> BERT initialized")
    # get and correct text
    def loadtext(self,filename,sesspath,force=True):
        text = self._load(filename,sesspath,force)
        text,text_original,incorrect = self._preprocess(text)
        text = self._correct(text,text_original,incorrect)
        os.remove(sesspath+"/tmp.pdf")
        os.remove(sesspath+"/tmp.txt")
        return text
    def _load(self,filename,sesspath,force):
        txt = sesspath+"/tmp.txt"
        ocr(sesspath+filename,sesspath+"/tmp.pdf",sidecar=txt,redo_ocr=(not force),deskew=force,rotate_pages=force,remove_background=force,clean=force,force_ocr=force)
        with open(sesspath+"/tmp.txt","r") as txt:
            text = txt.read()
        print("> OCR complete")
        return text
    # from Ravi Ilango's Medium Post
    def _preprocess(self,text):
        text = re.sub("\n\d+\n","",text)
        quotes = {'‘':"'", '’':"'", '“':'"', '”':'"','':'','-\n':''}
        quotes = dict((re.escape(k),v) for k,v in quotes.items())
        pattern = re.compile("|".join(quotes.keys()))
        text = pattern.sub(lambda m: quotes[re.escape(m.group(0))], text)
        text_original = text
        # cleanup text
        rep = { '\n': ' ', '\\': ' ', '\"': ' " ', '-': ' ', '|': ' | ',
                '[': ' [ ', ']': ' ] ', ',':' , ', '.':' . ', '!':' ! ',
                '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', '—':' ',
                '(': ' ( ', ')': ' ) ', "s'": "s '", ":":" : ",";":" ; "}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))
        text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
        text = re.sub("\d+"," ",text)
        text = re.sub("'(?!s )"," ' ",text)
        try:
            pos_tag(text.split()[0])
        except:
            import nltk
            nltk.download('averaged_perceptron_tagger')
        def get_personslist(text):
            tagged = set(pos_tag(text.split()))
            return list(set([word for word,pos in tagged if pos == 'NNP']))
        personslist = get_personslist(text)
        #print(personslist)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'","[","]","|",":",";"]
        # using pyspellchecker, identify incorrect words
        self.sc.word_frequency.load_words(ignorewords)
        self.sc.word_frequency.load_words(["pelargonium","pelargoniums"])
        words = text.split() #originally text.split()
        incorrectwords = [w for w in words if not list(self.sc.known([w])) and len(w) > 1]
        # replace incorrect words with [MASK]
        mask = self.tokenizer.mask_token
        for w in incorrectwords:
            text = text.replace(w+" ", mask, 1)
            #text_original = text_original.replace(w, mask)
        print(f"> {len(incorrectwords)} words to fix")
        print("> preprocessed text")
        return text,text_original,incorrectwords

    def _correct(self,text,text_original,incorrectwords):
        max_tokens = 512
        tokenized_text = self.tokenizer(text,return_tensors="pt")
        inids = tokenized_text["input_ids"]
        attid = tokenized_text["attention_mask"]
        normins = list(torch.split(inids,max_tokens,dim=1))
        normats = list(torch.split(attid,max_tokens,dim=1))
        print("> running BERT")
        for inz,atz in zip(normins,normats):
            toktxt = {"input_ids": inz,"attention_mask": atz}
            mask_token_ids = torch.where(toktxt["input_ids"] == self.tokenizer.mask_token_id)[1].tolist()
            if mask_token_ids == []:
                continue
            with torch.no_grad():
                token_logits = self.model(**toktxt).logits
            for token_id,word in zip(mask_token_ids,incorrectwords):
                mask_token_logits = token_logits[0,token_id,:]
                top_tokens = torch.topk(mask_token_logits,50).indices.tolist()
                choose = ""
                score = -1
                candidates = self.sc.candidates(word)
                for token in top_tokens:
                    unmask = self.tokenizer.decode(token)
                    mscore = string_metric(word,unmask)
                    for c in candidates:
                        mscore += string_metric(word,unmask)
                    if mscore > score and len(unmask)>1:
                        choose = unmask
                if choose.count("##") > 0:
                    replaced = text_original.replace(" "+word,choose.replace('##',''),1)
                    if replaced != text_original:
                        text_original = replaced
                    else:
                        text_original = text_original.replace(word,choose.replace('##',''),1)
                else:
                    text_original = text_original.replace(word,choose,1)
            s = slice(len(mask_token_ids),None)
            incorrectwords = incorrectwords[s]
        print("> finished BERT correction")
        text_original = text_original.replace("\n"," ")
        return text_original

