# -*- coding: utf-8 -*-

import csv
import pandas as pd
from pathlib import Path
import math

file_name = './data/reuter_dataset.tsv'
list_article_body = []
list_label = []

with open(file_name, encoding='utf-8', newline='') as f:
    for cols in csv.reader(f, delimiter='\t'):
        list_article_body.append(cols[0])
        list_label.append(cols[1])

df = pd.DataFrame({'text': list_article_body, 'label': list_label})
df = df.sample(frac=1, random_state=12).reset_index(drop=True)

len_0_2 = len(df) // 5

# 前から2割をテストデータとする
df[:len_0_2].to_csv("data/test.tsv", sep='\t', index=False, header=None)
print(df[:len_0_2].shape)

# 前2割からを訓練&検証データとする
df[len_0_2:len_0_2*3].to_csv("data/train.tsv", sep='\t', index=False, header=None)
print(df[len_0_2:len_0_2*3].shape)

df[len_0_2*4:].to_csv("data/eval.tsv", sep='\t', index=False, header=None)
print(df[len_0_2*4:].shape)


list_test_article_body = []
list_test_label = []
list_train_eval_article_body = []
list_train_eval_label = []
list_val_article_body = []
list_val_label = []

with open('./data/test.tsv', encoding='utf-8', newline='') as f:
    for cols in csv.reader(f, delimiter='\t'):
        list_test_article_body.append(cols[0])
        
        if float(cols[1]) > 0:
            cols[1] = 1
        else:
            cols[1] = 0

        list_test_label.append(cols[1])

with open('./data/train.tsv', encoding='utf-8', newline='') as f:
    for cols in csv.reader(f, delimiter='\t'):
        list_train_eval_article_body.append(cols[0])
        
        if float(cols[1]) > 0:
            cols[1] = 1
        else:
            cols[1] = 0
        list_train_eval_label.append(cols[1])

with open('./data/eval.tsv', encoding='utf-8', newline='') as f:
    for cols in csv.reader(f, delimiter='\t'):
        list_val_article_body.append(cols[0])
        
        if float(cols[1]) > 0:
            cols[1] = 1
        else:
            cols[1] = 0
        list_val_label.append(cols[1])

print('type(list_val_label[1]:',type(list_val_label[1]))

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


train_encodings = tokenizer(list_train_eval_article_body, truncation=True, padding=True)
val_encodings = tokenizer(list_val_article_body, truncation=True, padding=True)
test_encodings = tokenizer(list_test_article_body, truncation=True, padding=True)

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # encodingsはitemでdicをリストのようにKeyとValで扱える
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, list_train_eval_label)
val_dataset = IMDbDataset(val_encodings, list_val_label)
test_dataset = IMDbDataset(test_encodings, list_test_label)














from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    print(epoch,'回目')
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()