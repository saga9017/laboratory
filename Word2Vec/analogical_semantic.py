import numpy as np


f = open("saved_matrix/text8_skip-gram_negative-sampling", 'r')

dic={}
embeding_matrix=[]
i=0
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    dic[line[0]]=i
    word_matrix= [float(s) for s in line[1:]]
    embeding_matrix.append(word_matrix)
    i+=1

f.close()
reverse_dic = {y:x for x,y in dic.items()}

print(np.array(list(dic.keys())).shape)
word_dict=dic
index_dict=reverse_dic
W_in=np.array(embeding_matrix)
print(W_in.shape)





word_pair = open('questions-words.txt','r', encoding='utf-8')
word_line = []
for words in word_pair.readlines()[1:]:
    word_line.append(words)

pair = []
for words in word_line:
    if ':' in words:
        continue
    else:
        pair.append(words.split())

norm = np.sqrt(np.sum(np.square(W_in), 1, keepdims=True))
x = W_in / norm

score=0
mm=0
for word1,word2,word3,word4 in pair[:8869]:
    mm+=1
    if mm % 100 == 0:
        print(mm)
    if not word1 in word_dict:
        continue
    if not word2 in word_dict:
        continue
    if not word3 in word_dict:
        continue
    if not word4 in word_dict:
        continue
    test=W_in[word_dict[word2]]-W_in[word_dict[word4]]+W_in[word_dict[word3]]
    #test=W_in[word_dict[word]]

    norm2 = np.sqrt(np.sum(np.square(test)))
    y=test/norm2
#    distan=[]
#    for i in range(len(x)):
#        distan.append(np.sum(x[i]*y))
    distan=np.dot(x,y)
    seti=[]
    k=np.argsort(distan*np.array(-1))[:4]
    for i in k:
        seti.append(index_dict[i])

    if word1 in seti:
        score+=1

print(score)