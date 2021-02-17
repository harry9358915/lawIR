import json
import os
json_array = json.load(open("task1_train_2020_labels.json"))
subdirs = os.listdir("task1_clean_train_2020")
out = open("dev_id.txt"+'.tmp', 'w')
answers={}
with open("dev_id.txt","r") as file:
    for item in json_array:
        answers[item]=([int(os.path.splitext(i)[0]) for i in json_array[item]])
    for line in file.readlines():
        line1 = line.rstrip().split("\t")
        queryid = line1[0]
        docid = int(line1[1])
        if docid in answers[queryid]:
            out.write(line.rstrip()+"\t1\n")
        else:
            out.write(line.rstrip()+"\t0\n")
out.close()
os.remove("dev_id.txt")
os.rename("dev_id.txt"+'.tmp', "dev_id.txt")
