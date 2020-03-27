# skip-gram_hierarchical-softmax
import numpy as np
import random
import math
import time
import operator

# 파일을 열어 단어를 읽어 온다.
f = open("text8.txt", 'r')
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
n_gram=[3,4,5,6]
n_gram_dic={}

embeding_size = 100
window_size = 3

# learning rate
alpha = 0.025

# negative_sample의 개수
negative_sample = 20

# subsampling을 위한 t
t = 0.00001

# unknown으로 취급할 최소 개수
unknown_number = 50

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

n_gram_number=0
for x in index_of_word.keys():
    y='<'+x+'>'
    for j in n_gram:
        for i  in range(len(y)-j+1):
            if y[i:i+j] not in n_gram_dic.keys():
                n_gram_dic[y[i:i+j]]=n_gram_number
                n_gram_number += 1
    if y not in n_gram_dic.keys():
        n_gram_dic[y]=n_gram_number
        n_gram_number+=1

print('n_gram_number :', n_gram_number)

vocabulary_size = len(index_of_word)
print('vocabulary_size :', vocabulary_size)


dic_prob = {x: y / total_word_number for x, y in dic.items()}
subsampling_prob = {x: 1 - math.sqrt(t / y) for x, y in dic_prob.items()}
subsampled_table={}
for x, y in subsampling_prob.items():
    if y<0:
        subsampled_table[x]=[1]
    else:
        subsampled_table[x]=([0]*int(y*1000))+([1]*(1000-int(y*1000)))




# input, output 만들기
for i in range(len(words)):
    if words[i] not in dic.keys():
        words[i] = 'UNKNOWN'
    sampling_index=random.choice(subsampled_table[words[i]])
    if sampling_index == 1:
        if i < window_size:
            for j in range(window_size + i):
                input.append(words[i])
            for j in range(-window_size + i, window_size + 1):
                if j != 0 and i + j >= 0:
                    output.append(words[i + j])

        elif i >= window_size and i < len(words) - (window_size + 1):
            for j in range(window_size * 2):
                input.append(words[i])
            for j in range(-window_size, window_size + 1):
                if j != 0:
                    output.append(words[i + j])
        else:
            for j in range(len(words) - i + 1):
                input.append(words[i])
            for j in range(-window_size, len(words) - i):
                if j != 0:
                    output.append(words[i + j])

print('training sample의 수 :', len(input))
"""""
for i in range(32):
    print(input[i],'->',output[i])
"""""


def sigmoid(z):
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


w_in=np.random.uniform(-1,1,(n_gram_number,  embeding_size))
w_out=np.random.uniform(-1,1,(vocabulary_size, embeding_size)).T
print('w_in.shape :', w_in.shape)
print('w_out.shape :', w_out.shape)

table=table_for_negative_sampling()

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
    input_word = '<' + input_word + '>'
    divided_input_word = []
    for j in n_gram:
        for i in range(len(input_word) - j + 1):
            divided_input_word.append(input_word[i:i + j])
    if input_word not in divided_input_word:
        divided_input_word.append(input_word)

    input_word_numbers = [n_gram_dic[i] for i in divided_input_word]
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

    sum_w_in = np.sum(w_in[input_word_numbers], axis=0)

    y = softmax(np.dot(softmax_samples, sum_w_in))


    #y와 결과값을 계산
    e = y
    e[-1]-=1

    # 내적을 위해 e를 2차원 배열로 만듦
    e = np.reshape(e, (-1, 1))

    temp=np.reshape(np.dot(softmax_samples.T,e), -1)

    #w_out을 먼저 update 한다.
    j = 0
    for i in samples:
        w_out.T[i] -= alpha * sum_w_in * e[j]
        j += 1

    #w_out을 update한 뒤 w_in을 update 한다.
    # w_out을 update한 뒤 w_in을 update 한다.
    for i in input_word_numbers:
        w_in[i] -= alpha * temp



    index += 1

    if index % 100000 == 0:
        print('training!!! ', index * 100 / len(input), '% 완료')


print("training done!!!")

def save(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        y = '<' + word + '>'
        divided_input_word = []
        for j in n_gram:
            for i in range(len(y) - j + 1):
                divided_input_word.append(y[i:i + j])
        if y not in divided_input_word:
            divided_input_word.append(y)
        input_word_numbers = [n_gram_dic[i] for i in divided_input_word]
        sum_w_in = np.sum(w_in[input_word_numbers], axis=0)
        vector_str = ' '.join([str(s) for s in sum_w_in])
        f.write('%s %s\n' % (word, vector_str))

    f.close()
    print("저장 완료!!!")


def save_n_gram(file_name):
    f = open(file_name, 'w')
    for n_gram in list(n_gram_dic.keys()):
        n_gram_index = n_gram_dic[n_gram]
        vector_str = ' '.join([str(s) for s in w_in[n_gram_index]])
        f.write('%s %s\n' % (n_gram, vector_str))

    f.close()
    print("저장 완료!!!")


save('saved_matrix/fasttext_skip_neg_text8')
save_n_gram('saved_matrix/fasttext_skip_neg_text8_n')