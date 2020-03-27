import numpy as np
import random
import math
import time
import operator


f = open("text9.txt", 'r')
words = f.read().split(' ')
f.close()

# text에 있는 전체 단어의 수를 센다.
total_word_number = len(words)
print('단어 수 :', total_word_number)

# first_dic에는 (word: 그 단어수)의 dictionary를 부여한다.
prime_dic = {}

# dic에는 unknown이라는 단어를 포함한다.
dic = {}

# index_of_word에는 해당 단어의 index, 배열 내 위치 를 저장한다.
index_of_word = {}

embeding_size = 100
window_size = 5

# learning rate
alpha = 0.025

# negative_sample의 개수
negative_sample = 20

# subsampling을 위한 t
t = 0.00001

# unknown으로 취급할 최소 개수
unknown_number = 5

input = []
output = []

# 먼저 first_dic에 각 단어의 개수를 센다.
for word in words:
    if word != '':
        if word in prime_dic.keys():
            prime_dic[word] += 1
        else:
            prime_dic[word] = 1

# unknown_number이하로 등장하는 단어는 UNKNOWN으로 처리한다.
dic['UNKNOWN'] = 0
index_of_word['UNKNOWN'] = 0

index = 1
for x, y in prime_dic.items():
    if y < unknown_number:
        dic['UNKNOWN'] += y
    else:
        dic[x] = y
        index_of_word[x] = index
        index += 1

vocabulary_size = len(index_of_word)
dic_prob = {x: y / total_word_number for x, y in dic.items()}
print('vocabulary_size :', vocabulary_size)
print("dic :", dic)
word_of_index={y:x for x, y in index_of_word.items()}
print('word_of_index :', word_of_index)
#print(index_of_word)

cooccurrence_matrix=np.zeros((len(dic), len(dic)))
print('cooccurrence_matrix shape:', cooccurrence_matrix.shape)


# coocurrence_matrix의 생성
for i in range(len(words)):
    if words[i] not in dic.keys():
        words[i] = 'UNKNOWN'
    if i < window_size:
        for j in range(-window_size + i, window_size + 1):
            if j != 0 and i + j >= 0:
                if words[i+j] not in dic.keys():
                    words[i+j] = 'UNKNOWN'
                cooccurrence_matrix[index_of_word[words[i]]][index_of_word[words[i+j]]]+=1

    elif i >= window_size and i < len(words) - (window_size + 1):
        for j in range(-window_size, window_size + 1):
            if j != 0:
                if words[i+j] not in dic.keys():
                    words[i+j] = 'UNKNOWN'
                cooccurrence_matrix[index_of_word[words[i]]][index_of_word[words[i+j]]]+=1
    else:
        for j in range(-window_size, len(words) - i):
            if j != 0:
                if words[i+j] not in dic.keys():
                    words[i+j] = 'UNKNOWN'
                cooccurrence_matrix[index_of_word[words[i]]][index_of_word[words[i+j]]]+=1


print('cooccurrence_matrix :')
print(cooccurrence_matrix)

def save_cooccurrence_matrix(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        word_index = index_of_word[word]
        vector_str = ' '.join([str(s) for s in cooccurrence_matrix[word_index]])
        f.write('%s %s\n' % (word, vector_str))

    f.close()
    print("저장 완료!!!")

#save_cooccurrence_matrix('cooccurence_matrix')


pmi=np.zeros((len(dic), len(dic)))
for i in dic.keys():
    for j in dic.keys():
        x=index_of_word[i]
        y=index_of_word[j]
        if cooccurrence_matrix[y][x]==0:
            pmi[x][y]=0
        else:
            pmi[x][y]=np.log(cooccurrence_matrix[y][x]/np.sum(cooccurrence_matrix[y]))-np.log(dic_prob[i])


print('training done!!!')


def save(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        word_index = index_of_word[word]
        vector_str = ' '.join([str(s) for s in pmi[word_index]])
        f.write('%s %s\n' % (word, vector_str))

    f.close()
    print("저장 완료!!!")



save('saved_matrix/text9_pmi')
