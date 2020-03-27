import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


f = open("saved_matrix/text8_cbow_neg_vs.txt", 'r', encoding='utf-8')

dic={}
embeding_matrix=[]
i=0
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    dic[line[0]]=i
    word_matrix= [float(s) for s in line[1:] if s!='\n']
    embeding_matrix.append(word_matrix)
    i+=1

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


similar_word('father', 8)
