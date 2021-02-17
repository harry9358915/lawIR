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

subdirs = os.listdir("task1_clean_dev_2020")
root_path = "task1_clean_dev_2020/"
save_file_name = "dev_id.txt"
#all_letters = string.ascii_letters + " .,;'"

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        #and c in all_letters
    )

bm25_rank_file = open(save_file_name,mode="w")
for task_id in subdirs:
    print("task"+str(task_id))
    documents = []
    queries = []
    taskdir= root_path+task_id
    candidates_list = os.listdir(taskdir + '/candidates')
    docs_list=[]
    #pathlib.Path(new_root_path+ task_id+ '/candidates/').mkdir(parents=True, exist_ok=True)

    with open(root_path+ task_id+ "/base_case.txt", mode="r", encoding="utf-8") as file:
        query_list=[]
        for line in file.readlines():    
            line=line.rstrip()
            #line=unicodeToAscii(line)
            filtered_word_list = line.split()
            #filtered_word_list = [word.lower() for word in line if (( len(word) >= 3 and word[0].isalpha() and word.lower() not in stop_words ))]
            query_list.extend(filtered_word_list)
        docs_list.append(query_list)

    for candidatesfile in candidates_list:
        with open(root_path+task_id+ '/candidates/' + candidatesfile, mode="r", encoding="utf-8") as file:
            doc=[]
            for line in file.readlines():    
                line=line.rstrip()
                #line=unicodeToAscii(line)
                filtered_word_list = line.split()
                #filtered_word_list = [word.lower() for word in line if (( len(word) >= 3 and word[0].isalpha() and word.lower() not in stop_words ))]
                doc.extend(filtered_word_list)
        docs_list.append(doc)

    scores = get_bm25_weights(docs_list)
    scores[0] = scores[0][1:]
    best_docs = sorted(range(len(scores[0])), key=scores[0].__getitem__)
    best_docs.reverse()
    for rank,id1 in enumerate(best_docs[:100]):
        context = str(task_id)+"\t"+ str(id1+1) + "\t" + str(rank+1) + "\t" + str(scores[0][id1]) + "\n"
        print(context)
        bm25_rank_file.write(context)
    '''
    best_docs = sorted(range(len(scores[0])), key=scores[0].__getitem__)
    best_docs.reverse()
    best_docs.remove(0)
    bm25_rank_file.write(task_id+",")
    for rank,id1 in enumerate(best_docs[:100]):
        bm25_rank_file.write(str(id1)+" ")
    bm25_rank_file.write("\n")
    '''