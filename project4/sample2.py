import pickle
from transformers import *

with open('data.pickle', 'rb') as f:
    data = pickle.load(f)

print(len(data))

val_data=data[:1311]
test_data=data[1311:2622]
train_data=data[2622:]

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

new_val_loc_text_list = []

for idx, i in enumerate(val_data):
    print([idx]+tokenizer.encode(i[0]))
    new_val_loc_text_list.append([idx]+tokenizer.encode(i[0]))