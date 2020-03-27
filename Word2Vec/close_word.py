import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator

def similar_words_dic(dic, embeding_matrix, word, index):


    similarity_list=[]
    similarity_dic={}
    for i in list(dic.keys()):

        #cosine_similarity를 위해 (-1,1)로 reshape
        temp1=np.reshape(embeding_matrix[dic[word]], (1,-1))
        temp2=np.reshape(embeding_matrix[dic[i]], (1,-1))

        similarity_dic[i]=cosine_similarity(temp1, temp2)

    similarity_dic= sorted(similarity_dic.items(), key=operator.itemgetter(1))
    while True:
        if similarity_dic[-1][1]>index:
            similarity_list.append(similarity_dic[-1])
        else:
            break
        del similarity_dic[-1]
    return similarity_list

