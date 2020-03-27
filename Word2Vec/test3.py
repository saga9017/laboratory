import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator
import load_wordvector
from list_operation import sum, subtraction, multiply
from sklearn.cluster import KMeans

#원래 word2vec으로 학습된 data를 불러온다.
loaded_dic, loaded_matrix=load_wordvector.load_wordvector()


#분산된 data파일
f = open("polysemy", 'r')

dic={}
embeding_matrix=[]


while True:
    line = f.readline().split(' ')
    if not line[0]: break
    word_matrix = [float(s) for s in line[1:]]
    if line[0] not in dic.keys():
        dic[line[0]]=[len(dic), 1]
        embeding_matrix.append(word_matrix)
    else:
        dic[line[0]][1]+=1
        embeding_matrix[dic[line[0]][0]]=sum(word_matrix , embeding_matrix[dic[line[0]][0]])

"""""
#kmeans 알고리즘 적용
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    word_matrix = [float(s) for s in line[1:]]
    #line[0]는 단어를 가리킨다, dic은 [단어 index, 단어 수]를 mapping한다.
    if line[0] not in dic.keys():
        dic[line[0]]=[len(dic), 1]
        embeding_matrix.append([word_matrix])
    else:
        dic[line[0]][1]+=1
        embeding_matrix[dic[line[0]][0]]=[word_matrix]+embeding_matrix[dic[line[0]][0]]
f.close()


for x, y  in dic.items():
    X=embeding_matrix[y[0]]
    kmeans = KMeans(n_clusters=1, random_state=0).fit(X)
    #print(kmeans.inertia_)
    object_number=list(kmeans.labels_).count(0)
    #print(kmeans.cluster_centers_)
    if x not in loaded_dic.keys():
        embeding_matrix[y[0]] = sum(loaded_matrix[loaded_dic['UNKNOWN']],
                                    multiply(subtraction(kmeans.cluster_centers_, loaded_matrix[loaded_dic['UNKNOWN']]),
                                             object_number))
    else:
        embeding_matrix[y[0]] = sum(loaded_matrix[loaded_dic[x]],
                                    multiply(subtraction(kmeans.cluster_centers_, loaded_matrix[loaded_dic[x]]),
                                             object_number))
"""""




#reverse_dic = {y:x for x,y in dic.items()}

for x, y in dic.items():
    if x not in loaded_dic.keys():
        embeding_matrix[y[0]] = sum(loaded_matrix[loaded_dic['UNKNOWN']],
                                    subtraction(embeding_matrix[y[0]], multiply(loaded_matrix[loaded_dic['UNKNOWN']], y[1])))
    else:
        embeding_matrix[y[0]] = sum(loaded_matrix[loaded_dic[x]],
                                    subtraction(embeding_matrix[y[0]], multiply(loaded_matrix[loaded_dic[x]], y[1])))

print(np.array(list(dic.keys())).shape)

print(np.array(embeding_matrix).shape)

def similar_word(word, k):
    similarity_dic={}
    max_similarity = 0
    for i in list(dic.keys()):

        #cosine_similarity를 위해 (-1,1)로 reshape
        temp1=np.reshape(embeding_matrix[dic[word][0]], (1,-1))
        temp2=np.reshape(embeding_matrix[dic[i][0]], (1,-1))

        similarity_dic[i]=cosine_similarity(temp1, temp2)

    similarity_dic= sorted(similarity_dic.items(), key=operator.itemgetter(1))
    print(similarity_dic)
    for i in range(k):
        print(similarity_dic[-1])
        del similarity_dic[-1]



similar_word('three', 8)
similar_word('one', 8)
