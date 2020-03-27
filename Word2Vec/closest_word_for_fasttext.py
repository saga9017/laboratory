import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import operator


predict_word=['narrow-mindedness','department', 'campfires', 'knowing','urbanize','imperfection', 'principality', 'abnormal','secondary' ,'ungraceful']
f = open("saved_matrix/fasttext_skip_neg_text8", 'r')

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

f = open("saved_matrix/fasttext_skip_neg_text8_n", 'r')

n_gram_dic={}
n_gram_embeding_matrix=[]
i=0
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    n_gram_dic[line[0]]=i
    n_gram_matrix= [float(s) for s in line[1:]]
    n_gram_embeding_matrix.append(n_gram_matrix)
    i+=1

f.close()

print(np.array(list(dic.keys())).shape)
embeding_matrix=np.array(embeding_matrix)
n_gram_embeding_matrix=np.array(n_gram_embeding_matrix)
print(embeding_matrix.shape)

n_gram=[3,4,5,6]

def similar_word_in_vocab(word, k):
    similarity_dic={}
    temp1 = np.reshape(embeding_matrix[dic[word]], (1, -1))
    for i in list(dic.keys()):

        #cosine_similarity를 위해 (-1,1)로 reshape
        temp2=np.reshape(embeding_matrix[dic[i]], (1,-1))

        similarity_dic[i]=cosine_similarity(temp1, temp2)

    similarity_dic= sorted(similarity_dic.items(), key=operator.itemgetter(1))
    #print(similarity_dic)
    for i in range(k):
        print(similarity_dic[-1])
        del similarity_dic[-1]
    print('------------------------------------')

def similar_word_out_of_vocab(word, k):
    similarity_dic = {}
    if word=='narrow-mindedness':
        temp1 = np.reshape(embeding_matrix[dic['narrow']]+embeding_matrix[dic['minded']]+embeding_matrix[dic['ness']], (1, -1))

    elif word=='campfires':
        temp1 = np.reshape(embeding_matrix[dic['camp']]+embeding_matrix[dic['fire']]+n_gram_embeding_matrix[n_gram_dic['es>']], (1, -1))

    elif word=='urbanize':
        temp1 = np.reshape(embeding_matrix[dic['urban']]+n_gram_embeding_matrix[n_gram_dic['nize>']], (1, -1))

    elif word=='imperfection':
        temp1 = np.reshape(n_gram_embeding_matrix[n_gram_dic['<im']]+embeding_matrix[dic['perfect']]+n_gram_embeding_matrix[n_gram_dic['tion>']], (1, -1))


    elif word == 'ungraceful':
        temp1 = np.reshape(n_gram_embeding_matrix[n_gram_dic['<un']] + embeding_matrix[dic['grace']]+n_gram_embeding_matrix[n_gram_dic['ful>']], (1, -1))



    for i in list(dic.keys()):
        # cosine_similarity를 위해 (-1,1)로 reshape
        temp2 = np.reshape(embeding_matrix[dic[i]], (1, -1))

        similarity_dic[i] = cosine_similarity(temp1, temp2)

    similarity_dic = sorted(similarity_dic.items(), key=operator.itemgetter(1))
    print(word)
    for i in range(k):
        print(similarity_dic[-1])
        del similarity_dic[-1]

    print('------------------------------------')


for x in predict_word:
    if x in dic.keys():
        similar_word_in_vocab(x, 6)
    else:
        similar_word_out_of_vocab(x,5)

