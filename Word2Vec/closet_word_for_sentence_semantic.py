import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


f = open("test", 'r')

dic={}
embeding_matrix=[]
i=0
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    dic[(line[0], i%2)]=(i, line[1])
    word_matrix= [float(s) for s in line[2:]]
    embeding_matrix.append(word_matrix)
    i+=1

f.close()
reverse_dic = {y:x for x,y in dic.items()}

print(np.array(list(dic.keys())).shape)

print(np.array(embeding_matrix).shape)

def similar_word(word, index, top_k):
    similarity_dic={}
    max_similarity = 0
    # cosine_similarity를 위해 (-1,1)로 reshape
    temp1 = np.reshape(embeding_matrix[dic[(word,index)][0]], (1, -1))
    for i in list(dic.keys()):

        #cosine_similarity를 위해 (-1,1)로 reshape
        temp2=np.reshape(embeding_matrix[dic[i][0]], (1,-1))

        similarity_dic[i]=cosine_similarity(temp1, temp2)

    similarity_dic= sorted(similarity_dic.items(), key=operator.itemgetter(1))
    print(similarity_dic)
    for i in range(top_k):
        print(similarity_dic[-1])
        del similarity_dic[-1]


def word_distribution(word):
    print("--------------------------------")
    print((word, 0), ':' ,dic[(word,0)][1])
    print((word, 1), ':', dic[(word, 1)][1])




similar_word('apple', 1, 8)
word_distribution('apple')

#similar_word('one', 0, 8)
#similar_word('one', 1, 8)