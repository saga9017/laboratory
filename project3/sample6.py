
from transformers import *
import random

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

sen_len=5

f = open('train.txt', 'r', encoding='utf-8')
X_train = []
y_train = []
while True:
    line = f.readline()
    if not line: break
    temp= line.split('\t')

    X_train.append([len(X_train)]+[tokenizer.sep_token_id]+tokenizer.encode(temp[0].replace('\ufeff', ''))+[tokenizer.sep_token_id])
    y_train.append(int(temp[1].replace('\n', '').replace('\ufeff', '')))
f.close()

f = open('test.txt', 'r', encoding='utf-8')
X_test = []
y_test = []
while True:
    line = f.readline()
    if not line: break
    temp = line.split('\t')
    X_test.append([tokenizer.sep_token_id]+tokenizer.encode(temp[0].replace('\ufeff', ''))+[tokenizer.sep_token_id])
    y_test.append(int(temp[1].replace('\n', '').replace('\ufeff', '')))

f.close()

print(X_train)
print(y_train)

random.shuffle(X_train)
batch_train=[]
temp_X=[]
temp_X2=[]
temp_y=[]
temp_seg=[]
num_seen=0
max_len=0
for x in X_train:
    temp_X.append(x[1:])
    if max_len<len(x[1:]):
        max_len=len(x[1:])
    temp_y.append(y_train[x[0]])
    num_seen+=1

    if num_seen==5:
        for i in temp_X:
            temp_seg.append([1]*max_len)
            if len(i)<max_len:
                temp_X2.append(i+[tokenizer.pad_token_id] * (max_len - len(i)))
            else:
                temp_X2.append(i)

        batch_train.append((temp_X2, temp_seg, temp_y))
        temp_X = []
        temp_X2 = []
        temp_y = []
        temp_seg = []
        num_seen = 0
        max_len = 0

batch_test=[]
temp_X=[]
temp_X2=[]
temp_y=[]
temp_seg=[]
num_seen=0
max_len=0


for idx, x in enumerate(X_test):
    temp_X.append(x)
    if max_len<len(x):
        max_len=len(x)
    temp_y.append(y_test[idx])
    num_seen+=1

    if num_seen==5:
        for i in temp_X:
            temp_seg.append([1]*max_len)
            if len(i)<max_len:
                temp_X2.append(i+[tokenizer.pad_token_id] * (max_len - len(i)))
            else:
                temp_X2.append(i)

        batch_test.append((temp_X2, temp_seg, temp_y))
        temp_X = []
        temp_X2 = []
        temp_y = []
        temp_seg = []
        num_seen = 0
        max_len = 0
