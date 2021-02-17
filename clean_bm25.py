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

subdirs = os.listdir("task1_test_2020")
root_path = "task1_test_2020/"
new_root_path = "task1_clean_test_2020/"
#all_letters = string.ascii_letters + " .,;'"
bm25_rank_file = open("submission.txt",mode="w")
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        #and c in all_letters
    )

for task_id in subdirs:
    documents = []
    queries = []
    taskdir= root_path+task_id
    candidates_list = os.listdir(taskdir + '/candidates')
    docs_list=[]
    pathlib.Path(new_root_path+ task_id+ '/candidates/').mkdir(parents=True, exist_ok=True) 
    new_file = open(new_root_path +task_id+"/base_case.txt",mode="w", encoding="utf-8")
    with open(root_path + task_id+ "/base_case.txt",mode="r", encoding="utf-8") as file:
        query_list=[]
        for line in file.readlines():    
            line=line.rstrip()
            line=unicodeToAscii(line)
            line = re.split("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", line)
            filtered_word_list = [word.lower() for word in line if (( len(word) >= 3 and word[0].isalpha() and word.lower() not in stop_words ))]
            query_list.extend(filtered_word_list)
            if len(filtered_word_list) > 0:
                for item in filtered_word_list:
                    new_file.write("%s " % item)
                new_file.write("\n")
        docs_list.append(query_list)
    new_file.close()
    for candidatesfile in candidates_list:
        with open(root_path+task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
            with open(new_root_path+ task_id+ '/candidates/' + candidatesfile,mode="w", encoding="utf-8") as new_file:
                doc=[]
                for line in file.readlines():    
                    line=line.rstrip()
                    line=unicodeToAscii(line)
                    line = re.split("(?:(?:[^a-zA-Z]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)", line)
                    filtered_word_list = [word.lower() for word in line if (( len(word) >= 3 and word[0].isalpha() and word.lower() not in stop_words ))]
                    doc.extend(filtered_word_list)
                    if len(filtered_word_list)>0:
                        for item in filtered_word_list:
                            new_file.write("%s " % item)
                        new_file.write("\n")
        docs_list.append(doc)
    #print(query_list)
    '''
    scores = get_bm25_weights(docs_list)
    best_docs = sorted(range(len(scores[0])), key=scores[0].__getitem__)
    best_docs.reverse()
    best_docs.remove(0)
    bm25_rank_file.write(task_id+",")
    for rank,id1 in enumerate(best_docs[:100]):
        bm25_rank_file.write(str(id1)+" ")
    bm25_rank_file.write("\n")
    '''