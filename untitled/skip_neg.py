#skip-gram_negative-sampling
import numpy as np
import random
import math
import time
import operator

#파일을 열어 단어를 읽어 온다.
f = open("text8.txt", 'r')
words = f.read().split(' ')
f.close()

#text에 있는 전체 단어의 수를 센다.
total_word_number=len(words)
print('단어 수 :',total_word_number)

#first_dic에는 (word: 그 단어수)의 dictionary를 부여한다.
prime_dic={}

#dic에는 unknown이라는 단어를 포함한다.
dic={}

#index_of_word에는 해당 단어의 index, 배열 내 위치 를 저장한다.
index_of_word={}

embeding_size=100
window_size=2


#learning rate
alpha=0.025

#negative_sample의 개수
negative_sample=20

#subsampling을 위한 t
t=0.00001

#unknown으로 취급할 최소 개수
unknown_number=50

input=[]
output=[]

#먼저 first_dic에 각 단어의 개수를 센다.
for word in words:
    if word!='':
        if word in prime_dic.keys():
            prime_dic[word]+=1
        else:
            prime_dic[word]=1


#unknown_number이하로 등장하는 단어는 UNKNOWN으로 처리한다.
dic['UNKNOWN']=0
index_of_word['UNKNOWN'] = 0

index=1
for x,y in prime_dic.items():
    if y<unknown_number:
        dic['UNKNOWN']+=y
    else:
        dic[x]=y
        index_of_word[x] = index
        index+=1

vocabulary_size=len(index_of_word)
print('vocabulary_size :', vocabulary_size)
#print(index_of_word)


#input, output 만들기
for i in range(len(words)):
    if i<window_size:
        for j in range(window_size+i):
            input.append(words[i])
        for j in range(-window_size+i, window_size+1) :
            if j!=0 and i+j>=0:
                output.append(words[i+j])

    elif i>=window_size and i<len(words)-(window_size+1):
        for j in range(window_size*2):
            input.append(words[i])
        for j in range(-window_size, window_size+1) :
            if j!=0:
                output.append(words[i+j])
    else:
        for j in range(len(words)-i+1):
            input.append(words[i])
        for j in range(-window_size, len(words)-i) :
            if j!=0:
                output.append(words[i+j])

"""""
for i in range(32):
    print(input[i],'->',output[i])
"""""


#각 단어의 고유번호를 onehot vector로 만들기위한 함수
def one_hot_encoding(word):
    one_hot_vector = [0] * (vocabulary_size)
    index = index_of_word[word]
    one_hot_vector[index] = 1
    return one_hot_vector


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + np.exp(-z))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y



def table_for_negative_sampling():
    power = 0.75
    norm = sum([math.pow(dic[t], power) for t in list(dic.keys())])  # Normalizing constant

    table_size = 1e8  # Length of the unigram table
    table = np.zeros(int(table_size), dtype=np.int32)

    print('Filling unigram table')
    p = 0  # Cumulative probability
    i = 0
    for j, unigram in enumerate(list(dic.keys())):
        p += float(math.pow(dic[unigram], power)) / norm
        while i < table_size and float(i) / table_size < p:
            table[i] = j
            i += 1
    return table

dic_prob={x : y/total_word_number for x,y in dic.items()}
subsampling_prob={x : 1- math.sqrt(t/y) for x,y in dic_prob.items()}
#print('dic_prob :', dic_prob)
#print('subsampling_prob :', subsampling_prob)



def sample(table):
    indices = np.random.randint(low=0, high=len(table), size=negative_sample)
    return [table[i] for i in indices]



w_in=np.array([[random.uniform(-1,1) for i in range(embeding_size)] for j in range(vocabulary_size)])
w_out=np.array([[random.uniform(-1,1) for i in range(embeding_size)] for j in range(vocabulary_size)]).T
print('w_in.shape :', w_in.shape)
print('w_out.shape :', w_out.shape)

#negative sampling을 위해 unigram table을 채운다
print('unigram table 채우는 중!!')
table=table_for_negative_sampling()
print("unigram table 채우기 완료!!")


index = 0
# train하는 과정
for i in range(len(input)):
    # input, output 단어
    if input[i] not in dic.keys():
        input_word='UNKNOWN'
    else:
        input_word = input[i]

    if output[i] not in dic.keys():
        output_word='UNKNOWN'
    else:
        output_word = output[i]

    # 단어의 index
    input_word_number =index_of_word[input_word]
    output_word_number = index_of_word[output_word]

    #softmax할 idex를 가져온다
    samples=sample(table)

    #negative sampling을 했는데 target과 겹치는 것이 나오면 다시 sampling 한다.
    while True:
        if output_word in samples:
            samples=sample(table)
        else:
            break

    samples.append(output_word_number)

    #w_out에서 softmax를 취할 값들을 뽑는다.
    softmax_samples=np.array([w_out.T[i] for i in samples])

    y = softmax(np.dot(softmax_samples, w_in[input_word_number]))


    #y와 결과값을 계산
    e = y
    e[-1]-=1

    # 내적을 위해 e를 2차원 배열로 만듦
    e = np.reshape(e, (-1, 1))

    temp=np.reshape(np.dot(softmax_samples.T,e), -1)

    #w_out을 먼저 update 한다.
    j = 0
    for i in samples:
        w_out.T[i] -= alpha * w_in[input_word_number] * e[j]
        j += 1

    #w_out을 update한 뒤 w_in을 update 한다.
    w_in[input_word_number] -= alpha * temp



    index += 1

    if index % 100000 == 0:
        print('training!!! ', index * 100 / len(input), '% 완료')


print("training done!!!")



def save(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        word_index=index_of_word[word]
        vector_str = ' '.join([str(s) for s in w_in[word_index]])
        f.write('%s %s\n' % (word, vector_str))

    f.close()
    print("저장 완료!!!")


save('skip-gram_negative-sampling')
