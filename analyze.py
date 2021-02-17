import numpy as np
import os
import sys
import argparse
import json


if __name__ == "__main__":
    answers = {}
    queries = {}
    confusion_TP=0
    confusion_P=0
    confusion_precision=0
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str,default="task1_test_2020_labels.json",help="label")
    parser.add_argument("--submission", type=str,default="submission.txt",help="submission")
    args = parser.parse_args()
    json_array = json.load(open(args.label))
    for item in json_array:
        answers[item]=([int(os.path.splitext(i)[0]) for i in json_array[item]])
    with open(args.submission) as file:
        '''
        for line in file.readlines():
            query = [int(word)  for word in line.split(',')[1].split()[:-1]]
            queryid=line.split(',')[0]
            queries[queryid]=query
        '''
        query=[]
        for line in file.readlines():
            line = line.rstrip().split()
            query.append(int(line[1]))
            if len(query) == 10:
                queries[line[0]] = query
                query=[]
    for answers_key, answers_val in answers.items():
        queries_val=queries[answers_key]
        for item in answers_val:
            if item in queries_val:
                confusion_TP = confusion_TP + 1
        confusion_precision  = confusion_precision + len(queries_val)
        confusion_P = confusion_P + len(answers_val)
    
    print("Precision: ",confusion_TP/confusion_precision)
    print("Recall: ",confusion_TP/confusion_P)