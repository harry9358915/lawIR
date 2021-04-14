import numpy as np
import os
import sys
import argparse
import json
from collections import defaultdict

if __name__ == "__main__":
    queries = defaultdict(list)
    answers = {}
    confusion_TP=0
    confusion_P=0
    confusion_precision=0
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str,default="task1_test_2020_labels.json",help="label")
    parser.add_argument("--submission", type=str,default="test_result.tmp",help="submission")
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
        for line in file.readlines():
            line = line.rstrip().split()
            if float(line[3])>=0.4:
                queries[line[0]].append(int(line[1][3:]))

    confusion_FP=0
    confusion_TP=0
    confusion_FN=0
    for answers_key, answers_val in answers.items():
        queries_val=queries[answers_key]
        for item in queries_val:
            if item in answers_val:
                confusion_TP = confusion_TP + 1
                answers_val.remove(item)
            else:
                confusion_FP = confusion_FP + 1
        confusion_FN = confusion_FN + len(answers_val)
    Precision = confusion_TP/(confusion_TP+confusion_FP)
    Recall = confusion_TP/(confusion_TP+confusion_FN)
    F1score = 2*(Precision*Recall)/(Precision+Recall)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1-score: ", F1score)