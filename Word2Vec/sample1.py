
# Hierarchical Softmax
import os
import numpy as np
import time

os.chdir(r"C:\Users\Lee Wook Jin\PycharmProjects\Word2vec\training-monolingual.tokenized.shuffled")
before = time.time()
data_path = os.listdir()

word_dict = dict()
index_dict = dict()
freq_dict = dict()


for i in range(9):
    before = time.time()
    print(( i +1 ) *11 ,'/' ,len(data_path) ,' Data Loading.....')
    data =[]
    for path in data_path[ i *11:( i +1 ) *11]:
        edit = open(path ,'r' ,encoding='utf-8')
        for sentences in edit.readlines():
            data.append(sentences)

    after1 = time.time()
    print(( i +1 ) *11 ,'/' ,len(data_path) ,' Data Loading Finish   ', np.round(after1 -before ,2) ,'secs')

    print(( i +1 ) *11 ,'/' ,len(data_path) ,' Dictionary Loading.....')


    for sentence in data:
        tokens = sentence.split()
        for token in tokens:
            if not token in word_dict:
                word_dict[token] = len(word_dict)
                index_dict[len(index_dict)] = token
                freq_dict[token] = 1

            else:
                freq_dict[token] += 1

    after2 = time.time()
    print(( i +1 ) *11 ,'/' ,len(data_path) ,' Data Loading Finish   ', np.round(after2 -after1 ,2) ,'secs')

    del data



import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


f = open("saved_matrix/W_in_hier_e1.txt", 'r')

dic={}
embeding_matrix=[]
i=0
while True:
    line = f.readline().split('\t')
    for j in line:
        embeding_matrix.append(j.split(','))
    if not line[0]: break


f.close()
reverse_dic = {y:x for x,y in dic.items()}

print(np.array(list(dic.keys())).shape)

print(np.array(embeding_matrix).shape)

n_gram=[3,4,5,6]

def similar_word(word, k):
    similarity_dic={}
    max_similarity = 0
    for i in list(dic.keys()):

        #cosine_similarity를 위해 (-1,1)로 reshape
        temp1=np.reshape(embeding_matrix[dic[word]], (1,-1))
        temp2=np.reshape(embeding_matrix[dic[i]], (1,-1))

        similarity_dic[i]=cosine_similarity(temp1, temp2)

    similarity_dic= sorted(similarity_dic.items(), key=operator.itemgetter(1))
    print(similarity_dic)
    for i in range(k):
        print(similarity_dic[-1])
        del similarity_dic[-1]


similar_word('billion', 8)
