import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


f = open("C:/Users/Lee Wook Jin/source/repos/Fasttext_embedding/fasttest_neg5_sub_vs_1.txt", 'r', encoding='utf-8')

word_to_id={}
word_to_count={}
hidden_weights=[]
i=0
while True:
    line = f.readline().split(' ')
    if len(line)!=303: break
    word_to_id[line[0]]=i
    word_to_count[line[0]]=int(line[1])
    word_matrix= [float(s) for s in line[2:] if s!='\n']
    hidden_weights.append(word_matrix)
    i+=1

f.close()


# 평가
test_data = open(r'Efficient Estimation of Word Representations in Vector Space dataset.txt','r', encoding='utf-8')
raw = []
for lines in test_data.readlines()[1:]:
    raw.append(lines)

test_pair = []
for lines in raw:
    if ':' in lines:
        continue
    else:
        test_pair.append(lines.split())

semantic = test_pair[:8869]
syntatic = test_pair[8869:]

wor = {}
dd = {}
for key in word_to_id.keys():
    if word_to_count[key] > 1000:  # 30000개가 됨
        wor[key] = len(wor)
        dd[len(dd)] = key

len(wor)

input_weight = []

for word in wor:
    if word in word_to_id:
        input_weight.append(hidden_weights[word_to_id[word]])
    else:
        input_weight.append([-0.000000001] * 300)
input_weight = np.array(input_weight)

norm_all = np.sqrt(np.sum(np.square(input_weight), 1, keepdims=True))
all_ = input_weight / norm_all

hidden_weights=np.array(hidden_weights)
def OneEval(word2, word3, word4, all_, word_to_id):
    testing = hidden_weights[word_to_id[word2.lower()]] \
              - hidden_weights[word_to_id[word4.lower()]] \
              + hidden_weights[word_to_id[word3.lower()]]

    norm_testing = np.sqrt(np.sum(np.square(testing)))
    test = testing / norm_testing

    Cosine = np.dot(all_, test)

    sorting = np.argsort(Cosine * np.array(-1))[:4]
    top_word = []
    for top_ in sorting:
        top_word.append(dd[top_])

    print(top_word)


for word1, word2, word3, word4 in semantic[:5]:
    print(word1, '/', word2, word3, word4)
    OneEval(word2, word3, word4, all_, word_to_id)


def Eval(name, pair_data, all_, word_to_id):
    score = 0
    not_ = 0

    running = 0

    for word1, word2, word3, word4 in pair_data:
        running += 1
        if word1 not in word_to_id:
            not_ += 1
            continue

        if word2 not in word_to_id:
            not_ += 1
            continue

        if word3 not in word_to_id:
            not_ += 1
            continue

        if word4 not in word_to_id:
            not_ += 1
            continue

        testing = hidden_weights[word_to_id[word2.lower()]] \
                  - hidden_weights[word_to_id[word1.lower()]] \
                  + hidden_weights[word_to_id[word3.lower()]]

        norm_testing = np.sqrt(np.sum(np.square(testing)))
        test = testing / norm_testing

        Cosine = np.dot(all_, test)

        sorting = np.argsort(Cosine * np.array(-1))[:4]
        top_word = []
        for top_ in sorting:
            top_word.append(dd[top_])

        if word4 in top_word:
            score += 1

    print(" %s Test - %03f %%" % (name, score / len(pair_data) * 100))
    print("    -> CAN'T TESTING (NOT WORD) :", not_)
    print("    -> Adjusting Test : %03f %%" % ((score) / (len(pair_data) - not_) * 100))


Eval("Semantic", semantic, all_, word_to_id)
Eval("Syntatic", syntatic, all_, word_to_id)