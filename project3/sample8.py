import pickle
from transformers import *
import numpy as np

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

def load_k():
    k_train, k_val, k_test = [], [], []
    with open("project3_data/txt/e5_train_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_train.append(line)

    with open("project3_data/txt/e5_val_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_val.append(line)

    with open("project3_data/txt/e5_test_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_test.append(line)

    return k_train, k_val, k_test

with open("project3_data/vocabulary_keras_h.pkl", "rb") as f:
    data = pickle.load(f)
vocabulary = data[0]
hashtagVoc = data[2]
vocabulary_inv = {}
hashtagVoc_inv = {}
hashtagCount = {}
for k, v in vocabulary.items():
    vocabulary[k] = v + 2

vocabulary["<Padding>"] = 0
vocabulary['<CLS>'] = 1
vocabulary['<SEP>'] = 2

for i in vocabulary.keys():
    vocabulary_inv[vocabulary[i]] = i
for i in hashtagVoc.keys():
    hashtagVoc_inv[hashtagVoc[i]] = i
    hashtagCount[hashtagVoc[i]] = []

print("vocabulary 스펙:", len(vocabulary), max(vocabulary.values()), min(vocabulary.values()))
print("hashtagVoc 스펙 :", len(hashtagVoc), max(hashtagVoc.values()), min(hashtagVoc.values()))
print("len(hashtagVoc_inv)", len(hashtagVoc_inv))

# Knowledge-base 추가
k_train, k_val, k_test = load_k()
print(len(k_train), len(k_val), len(k_test))
################################# for padding, cls, sep ########################################################
for index, categories in enumerate(k_train):
    temp_category = []
    for category in categories:
        temp_category.append(category + 2)
    temp_category.append(2)
    k_train[index] = temp_category

for index, categories in enumerate(k_val):
    temp_category = []
    for category in categories:
        temp_category.append(category + 2)

    temp_category.append(2)
    k_val[index] = temp_category

for index, categories in enumerate(k_test):
    temp_category = []
    for category in categories:
        temp_category.append(category + 2)

    temp_category.append(2)
    k_test[index] = temp_category

####################################################################################################################


test_data = []
test_data.append(np.load("project3_transformer_data/transformer_image_90_test.npy"))
print("test data loading finished.")
with open("project3_data/test_tlh_keras_h.bin", "rb") as f:
    test_data.extend(pickle.load(f))
print("test data 업로드")

new_test_text_list = []
for index, sentece in enumerate(test_data[1]):
    temp_sen_ = []
    for word in sentece:
        if word == 0:
            pass
        else:
            temp_sen_.append(word + 2)
    temp_sen_ =[tokenizer.sep_token_id]+ tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen_]))+[tokenizer.sep_token_id]
    new_test_text_list.append(temp_sen_)

new_test_loc_list = []
for index, sentece in enumerate(test_data[2][:, -17:]):
    temp_sen = []
    for word in sentece:
        if word == 0:
            pass
        else:
            temp_sen.append(word + 2)
    temp_sen=[tokenizer.cls_token_id]+tokenizer.encode(' '.join([vocabulary_inv[j] for j in temp_sen]))+[tokenizer.sep_token_id]
    temp_sen.insert(0, index)
    new_test_loc_list.append(temp_sen)

new_test_loc_text_list = []

for x, y in zip(new_test_loc_list, new_test_text_list):
    new_test_loc_text_list.append([x[0]]+y)


test_data.append(k_test)

print(len(test_data[0]), len(test_data[1]), len(test_data[2]), len(test_data[3]), len(test_data[4]))
# test_data = check_hashzero(test_data)
# print("check 완")

print(len(new_test_loc_text_list))
print(len(new_test_loc_list))
print(len(test_data))

new_train_loc_text_list=new_test_loc_text_list[:5000]
new_train_loc_list=new_test_loc_list[:5000]
train_data=[]
train_data.append(test_data[0][:5000])
train_data.append(test_data[1][:5000])
train_data.append(test_data[2][:5000])
train_data.append(test_data[3][:5000])
train_data.append(test_data[4][:5000])

temp_loc_text_list=new_test_loc_text_list[5000:]
temp_loc_list=new_test_loc_list[5000:]
temp_data=[]
temp_data.append(test_data[0][5000:])
temp_data.append(test_data[1][5000:])
temp_data.append(test_data[2][5000:])
temp_data.append(test_data[3][5000:])
temp_data.append(test_data[4][5000:])

new_test_loc_text_list=temp_loc_text_list
new_test_loc_list=temp_loc_list
test_data=temp_data



############################################################################
#batch_test = generate_batch(new_test_loc_text_list, new_test_loc_list, test_data, len(hashtagVoc))
############################################################################