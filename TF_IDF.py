#from gensim import tfidfvectorizer
from collections import defaultdict
import numpy as np
import pandas as pd
import spacy
import json
import re
import string
from pathlib import Path
from pygments.lexers import guess_lexer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

class tfidf:
    def __init__(self):
        filepath = Path("Article")
        self.filename = [file for file in filepath.iterdir()]
        self.pattern = string.punctuation + "、。・￥「」『』［］【】！”＃＄％＆’（）＝～＾＠‘＊｛｝｜＿／\\\\"
        self.url = f"https://[{string.punctuation}a-zA-Z]"
        self.corpus = None
        for file in self.filename:
            df = pd.read_csv(file, encoding='utf-8')
            if self.corpus is None:
                self.corpus = df
            else:
                self.corpus = pd.concat((self.corpus, df), axis=0)
        self.nlp = spacy.load("ja_core_news_sm")
        self.text_splitter = RecursiveCharacterTextSplitter(
                                                  chunk_size=2000,
                                                  chunk_overlap=0)
        self.pre_corpus = self.corpus.body.copy()
        self.pre_corpus = self.pre_corpus.apply(lambda x: self.preprocess(x))
        
    
    def preprocess(self, doc): 
        prepro_text = "" #前処理後のテキスト
        doc = re.sub(self.url, "", doc)
        doc = re.sub(f"[{self.patter}]", " ", doc)
        doc = re.sub("[0-9０-９]+", "", doc)
        doc = doc.lower()
        doc = doc.split("\n")
        if_code = "" #ソースコード判定用の文字列
        check = False #ソースコードチェック用のbool値
        for text in doc:
            len_ja = len(re.findall("[ぁ-んァ-ヶー-龥々]", text))
            len_en = len(re.findall("[a-zA-Z]", text))
            if len_ja > len_en:
                if check:
                    lexer = guess_lexer(if_code)
                    if lexer.name in ["Text only", "Text output"]:
                        prepro_texts += if_code
                prepro_text += text
            else:
                flag = True
                if_code += text
        return prepro_text
    
    def dictionary(self):
        langdict = defaultdict(int)
        for doc in self.pre_corpus:
            split_texts = self.text_splitter.split_text(doc)
            for split_text in split_texts:
                words = self.nlp(split_text)
                for word in words:
                    if word.pos_ in ["NOUN", "PROPN"] and not word.is_stop:
                        langdict[word.lemma_.lower()] += 1
        langdict = sorted(langdict.items(), key=lambda x: x[1])
        langdict = dict(langdict)        
        return langdict
    
    def wordvalue(self, words):
        self.nounsum = sum([v for v in words.values()])
        self.wordvalues = dict()
        for k,v in words.items():
            word_tf = self.tf(v)
            self.wordvalues[k] = word_tf
        return self.wordvalues
    
    def tf(self, count):
        return count / self.nounsum
    
    def idf(self):
        return

    def guess_difficulty(self):
        result_sers = np.zeros(len(self.corpus))
        id = 0
        for doc in self.pre_corpus:
            result = []
            split_texts = self.text_splitter.split_text(doc)
            for split_text in split_texts:
                words = self.nlp(split_text)
                for word in words:
                    if word.pos_ in ["NOUN", "PROPN"] and not word.is_stop:
                        result.append(word.lemma_.lower())
            for word in result:
                result_sers[id] += self.wordvalues[word]
            id += 1
        result_sers = pd.Series(result_sers, name="difficulty")
        #result_sers = pd.concat((self.corpus.body, result_sers), axis=1)
        #result_sers = result_sers.sort_values(key="difficulty")
        return result_sers

            

        
                            