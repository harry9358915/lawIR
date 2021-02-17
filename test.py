import os
import unicodedata
import string
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.summarization.bm25 import get_bm25_weights
import pathlib


lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
dict_words = set(nltk.corpus.words.words())

subdirs = os.listdir("task1_clean_train_bert_2020")
root_path = "task1_clean_train_bert_2020/"
#all_letters = string.ascii_letters + " .,;'"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        #and c in all_letters
    )

bm25_rank_file = open("submission.txt",mode="w")
for task_id in subdirs:
    print("task"+str(task_id))
    candidates_list = os.listdir(root_path+ task_id + '/candidates')
    docs_list=[]
    queries={}
    docs={}
    #pathlib.Path(new_root_path+ task_id+ '/candidates/').mkdir(parents=True, exist_ok=True)
    with open(root_path+ task_id+ "/clean_base_case.txt", mode="r", encoding="utf-8") as file:
        for line in file.readlines():
            queries[task_id] = line
    for candidatesfile in candidates_list:
        with open(root_path+ task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
            context = file.read()
            context = " ".join(context.split())
            docs[task_id+os.path.splitext(candidatesfile)[0]] = context
    with open("train_id.txt",'r') as file:
        #self._examples = []
        for i,line in enumerate(file):
            line = line.strip().split("\t")
            line[1] = "{:0>3d}".format(int(line[1]))
            line[1] = line[0]+line[1]
            print(line[1])
        break
    break
print(queries)
print(docs["001002"])