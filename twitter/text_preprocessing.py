import os
import pickle

text_list = os.listdir('data_multimodal/text')

multi_modal_dataset = {}

for text_ in text_list:
    path = 'data_multimodal/text/' + str(text_)
    with open(path, 'r', encoding = 'utf-8') as f:
        texts = f.readlines()
    for text in texts:
        tmp = []
        tmp.append(text[24:-6])
        tmp.append(int(text[-3]))
        multi_modal_dataset[text[2:20]] = tmp


with open(r'data_multimodal/extract/extract_all', 'r', encoding='utf-8') as f:
    extract_data = f.readlines()

for extract in extract_data:
    key = extract[2:20]
    tmp1 = extract[24:-3].replace("', '", ' ')
    if key not in multi_modal_dataset:
        continue
    tmp = multi_modal_dataset[key]
    tmp.append(tmp1)
    multi_modal_dataset[key] = tmp

with open('textnextract.pickle', 'wb') as f:
    pickle.dump(multi_modal_dataset, f)

dic_keys = list(multi_modal_dataset.keys())

