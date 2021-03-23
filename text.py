from ocrmypdf import ocr
import torch
import re
import nltk
from enchant.checker import SpellChecker
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModel, pipeline
from pdftotext import PDF
import os
import numpy as np

nltk.download('popular')
print("> NLTK setup")

def _need_ocr(inpdf,minwords):
    with open(inpdf,"rb") as f:
        pdf = PDF(f)
    text = ' '.join(pdf)
    word_list = text.replace(',','').replace('\'','').replace('.','').lower().split()
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
    rep = { '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
                    '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
                            '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
                                    '(': ' ( ', ')': ' ) ', "s'": "s '"}
    rep = dict((re.escape(k),v) for k,v in rep.items())
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))],text)


class TP2:
    def __init__(self,sc_lang="en_US"):
        self.unmask = pipeline("fill-mask")
        self.sc = SpellChecker(sc_lang)
        print("> language completion model loaded")
    def loadtext(self,inpdf,path,minwords=10,ocr_lang=['eng'],unpaper_args=None,remove_citations=True):
        print(f"> loading {inpdf}")
        ocrtxt, pretxt = _get_text(inpdf,path,ocr_lang,unpaper_args,remove_citations)
        os.remove(f'{path}/tmp.pdf')
        os.remove(f'{path}/tmp.txt')
        #os.remove(inpdf)
        txts = [ocrtxt] + [pretxt] if pretxt != "" else []
        print(f"> processing {len(txts)}")
        for txt in txts:
            ft,ot,sw = self._preprocess(txt,remove_citations)
            txt = self._correct_words(ft,ot,sw)
        if len(txts)>1:
            return txts[0], txts[1]
        return txts[0]
    def _preprocess(self,text,remove_citations): 
        if remove_citations:
            text = _remove_citations(text)
        text = text.replace('...',';')
        text = text.replace('. . .',';')
        original_text = text
        text = _cleanup(text)
        personslist = _get_personslist(text)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]
        words = text.split()
        incorrectwords = [" "+w+" " for w in words if not self.sc.check(w) and w not in ignorewords]
        suggestedwords = [self.sc.suggest(w) for w in incorrectwords]
        for w in incorrectwords:
            text = text.replace(w, self.unmask.tokenizer.mask_token)
            original_text = original_text.replace(w, self.unmask.tokenizer.mask_token)
        return text, original_text, suggestedwords
    def _correct_words(self,txt,otxt,sw):
        print(f"> predicting {len(sw)} words")
        predictions = self.unmask(txt)
        for i,pred in enumerate(predictions):
            preds = [p.token_str for p in pred]
            suggs = sw[i]
            simmax = 0
            out=''
            for p in preds:
                for s in suggs:
                    r = SequenceMatcher(None,p,s).ratio()
                    if r is not None and r > simmax:
                        simmax = r
                        out = p
            otxt = otxt.replace(self.unmask.tokenizer.mask_token,out,1)
        return otxt

class TextProcessor(object):
    def __init__(self,sc_language="en_US",bert_model="distilbert-base-uncased"):
        self.sc = SpellChecker(sc_language)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.model = AutoModel.from_pretrained(bert_model)
        print("> BERT loaded")

    # take input pdf (inpdf) plus a bunch of settings and output a the fixed output (of both the new ocr and existing text layer)
    def loadtext(self,inpdf,sesspath,minwords=10,ocr_lang=['eng'],unpaper_args=None,remove_citations=True):
        print(f"> loading {inpdf}")
        textocr, pretext = _get_text(inpdf,sesspath,ocr_lang,unpaper_args,minwords)
        print(textocr[100:150],"---",pretext[100:150])
        os.remove(f'{sesspath}/tmp.pdf')
        os.remove(f'{sesspath}/tmp.txt')
        #os.remove(inpdf)
        texts = [textocr] + [pretext] if pretext is not None else []
        print(f"> processing {len(texts)}")
        for text in texts:
            ft,ot,sw = self._preprocess(text,remove_citations)
            text = self._predict_words(ft,ot,sw)
        if len(texts) == 1:
            return text[0]
        return text[0], text[1]

    def _preprocess(self,text,remove_citations): 
        if remove_citations:
            text = self._remove_citations(text)
        text = text.replace('...',';')
        text = text.replace('. . .',';')
        original_text = text#.copy()
        text = self._cleanup(text)
        personslist = self._get_personslist(text)
        ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]
        words = text.split()
        incorrectwords = [w for w in words if not self.sc.check(w) and w not in ignorewords]
        suggestedwords = [self.sc.suggest(w) for w in incorrectwords]
        for w in incorrectwords:
            text = text.replace(w, '[MASK]')
            original_text = original_text.replace(w, '[MASK]')
        return text, original_text, suggestedwords

    def _predict_words(self,text_filtered, text_original, suggestedwords):
        print(f"> predicting {len(suggestedwords)} words")
        # BERT time
        tokenized_text = self.tokenizer.tokenize(text_filtered)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        maskids = [i for i,e in enumerate(tokenized_text) if e == '[MASK]']
        # segment tensors
        segs =    [i for i,e in enumerate(tokenized_text) if e == '.']
        segids = []
        prev = -1
        for k,s in enumerate(segs):
            segids=segids+[k]*(s-prev)
            prev=s
        segids=segids+[len(segs)]*(len(tokenized_text)-len(segids))
        seg_tensors=torch.tensor([segids])
        # prep inputs
        tokens_tensor = torch.tensor([indexed_tokens])
        with torch.no_grad():
            predictions = self.model(tokens_tensor, seg_tensors)
        # refine BERT predictions with spellcheck
        print(predictions)
        for i in range(len(maskids)):
            preds = torch.topk(predictions[0][maskids[i]],k=50)
            indices = [j for l in preds.indices.tolist() for j in l]
            #print(indices)
            list1 = self.tokenizer.convert_ids_to_tokens(indices)
            list2 = suggestedwords[i]
            simmax=0
            predicted_token=''
            for word1 in list1:
                for word2 in list2:
                    s = SequenceMatcher(None,word1,word2).ratio()
                    if s is not None and s > simmax:
                        simmax = s
                        predicted_token = word1
            text_original = text_original.replace('[MASK]',predicted_token,1)
        return text_original
