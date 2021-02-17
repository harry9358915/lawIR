'''
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
print(inputs)
outputs = model(**inputs, labels=labels)
loss = outputs.loss
logits = outputs.logits
print(logits)
'''
scores={'123':0.5,'121':0.9}
a=sorted(scores.items(),key=lambda x: x[1], reverse=True)
print(a)