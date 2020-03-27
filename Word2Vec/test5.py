# skip-gram_hierarchical-softmax
import numpy as np
import random
import math
import time
import operator
from load_wordvector import load_wordvector
from sklearn.metrics.pairwise import cosine_similarity
from load_stopwords import load_stopwords

# 파일을 열어 단어를 읽어 온다.
f = open("text8.txt", 'r')
while True:
    line = f.readline()
    if not line: break
    words=line.split(' ')
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
learning_distribution={}

embeding_size = 100
window_size = 5

# learning rate
alpha = 0.025

# unknown으로 취급할 최소 개수
unknown_number = 50

input = []
outputs = []

#최대 의미의 수
meaning=5

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
learning_distribution['UNKNOWN']=[0]*meaning

index = 1
for x, y in prime_dic.items():
    if y < unknown_number:
        dic['UNKNOWN'] += y
    else:
        dic[x] = y
        learning_distribution[x]=[0]*meaning
        index_of_word[x] = index
        index += 1

vocabulary_size = len(index_of_word)
print('vocabulary_size :', vocabulary_size)
#print('index_of_word :',index_of_word)

#input, output 만들기
for i in range(len(words)):
    if words[i]!='tree':
        continue
    output_temp=[]
    if i<window_size:
        input.append(words[i])
        for j in range(-window_size+i, window_size+1) :
            if j!=0 and i+j>=0:
                output_temp.append(words[i+j])
        outputs.append(output_temp)
    elif i>=window_size and i<len(words)-(window_size+1):
        input.append(words[i])
        for j in range(-window_size, window_size+1) :
            if j!=0:
                output_temp.append(words[i+j])
        outputs.append(output_temp)
    else:
        input.append(words[i])
        for j in range(-window_size, len(words)-i) :
            if j!=0:
                output_temp.append(words[i+j])
        outputs.append(output_temp)


for i in range(32):
    print(input[i],'->',outputs[i])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


dic_prob = {x: y / total_word_number for x, y in dic.items()}
#print('dic_prob :', dic_prob)


def lowest_prob_pair(p):
    assert (len(p) >= 2)  # Ensure there are at least 2 symbols in the dist.
    sorted_p = sorted(p.items(), key=operator.itemgetter(1))
    return sorted_p[0], sorted_p[1]

def parent(self,list):
    if self.parent==None:
        return None
    list.append(self.parent.index)
    parent(self.parent, list)


class Node(object):
    def __init__(self, data):
        self.index=None
        self.parent=None


#huffman code의 생성
huffman_code=dic_prob.copy()

nodes={}
node_index=-1

print("huffman code 생성중!!")

while len(huffman_code.keys())!=1:
    a1, a2 = lowest_prob_pair(huffman_code)
    huffman_code[(a1[0], a2[0])] = a1[1] + a2[1]
    del huffman_code[a1[0]], huffman_code[a2[0]]
    nodes[(a1[0], a2[0])]=Node((a1[0], a2[0]))
    node=nodes[(a1[0], a2[0])]
    node_index += 1
    node.index = node_index

    if a1[0] in nodes.keys():
        nodes[a1[0]].parent=node
    else:
        nodes[a1[0]]=Node(a1[0])
        nodes[a1[0]].parent = node

    if a2[0] in nodes.keys():
        nodes[a2[0]].parent = node
    else:
        nodes[a2[0]] = Node(a2[0])
        nodes[a2[0]].parent = node



root=list(huffman_code.keys())[0]
root_node=nodes[root]
huffman_code[root]=''
temp={}


pivot=root
return_pivots=[]
while len(huffman_code.keys())<len(dic_prob.keys()):
    #print(len(huffman_code.keys()))
    huffman_code[pivot[0]]=huffman_code[pivot]+'0'
    huffman_code[pivot[1]]=huffman_code[pivot]+'1'

    del huffman_code[pivot]

    if type(pivot[1]) == tuple:
        return_pivots.append(pivot[1])

    if type(pivot[0])==tuple:
        pivot=pivot[0]

    else:
        if return_pivots==[]:
            pivot=None
        else:
            pivot=return_pivots.pop()


#print('huffman_code :', huffman_code)
print('huffman_code 구성 완료!!')


#word2vec으로 training한 vector들을 불러온다
loaded_dic, loaded_matrix=load_wordvector()
stopwords_list=load_stopwords()
stopwords_list.append('UNKNOWN')
w_in=np.random.uniform(-1,1,(meaning, vocabulary_size,  embeding_size))
w_out=np.random.uniform(-1,1,(node_index+1, embeding_size)).T
word_semantics=np.zeros((meaning, vocabulary_size,  embeding_size))
print('w_in.shape :', w_in.shape)
print('w_out.shape :', w_out.shape)
print('word_semantics.shape :', word_semantics.shape)


def in_train(input_word, output_word, w_in_index):
    # 단어의 index
    input_word_number = index_of_word[input_word]
    output_word_number = index_of_word[output_word]

    # output단어의 huffman code
    output_huffman_code = huffman_code[output_word]
    active_node = []
    parent(nodes[output_word], active_node)
    active_node.reverse()

    hierarchi_samples = np.array([w_out.T[i] for i in active_node])

    y = sigmoid(np.dot(hierarchi_samples, w_in_index[input_word_number]))

    j = 0
    e = np.zeros(y.shape)
    for i in output_huffman_code:
        if i == '0':
            e[j] = y[j]
        else:
            e[j] = y[j] - 1
        j += 1

    # 내적을 위해 e를 2차원 배열로 만듦
    e = np.reshape(e, (-1, 1))

    temp = np.reshape(np.dot(hierarchi_samples.T, e), -1)

    # w_out을 먼저 update 한다.
    j = 0
    for i in active_node:
        w_out.T[i] -= alpha * w_in_index[input_word_number] * e[j]
        j += 1

    # w_out을 update한 뒤 w_in을 update 한다.
    w_in_index[input_word_number] -= alpha * temp



step = 0
# train하는 과정
for i in range(len(input)):
    # input, output 단어
    output_words = []

    if input[i] not in dic.keys():
        input_word = 'UNKNOWN'
    else:
        input_word = input[i]

    for j in outputs[i]:
        if j not in dic.keys():
            output_words.append('UNKNOWN')
        else:
            output_words.append(j)

    sentence_semantic=[]
    for j in output_words:
        if j not in stopwords_list:
            sentence_semantic.append(loaded_matrix[loaded_dic[j]])
            print(j, end=' ')
    print()
    sentence_semantic=np.sum(sentence_semantic, axis=0)
    if sentence_semantic.shape==tuple():
        continue
    #semantic_0의 0의 개수, 의미가 최대 다섯가지인 경우
    similarity_list=[]

    for j in range(meaning):
        word_semantic=word_semantics[j][index_of_word[input_word]]
        if np.count_nonzero(word_semantic==0)==embeding_size:
            pivot=j
            break
        else:
            similarity = cosine_similarity(sentence_semantic.reshape(1, -1), word_semantic.reshape(1, -1))
            similarity_list.append(similarity)
    print('similarity_list :', similarity_list)

    if pivot==0:
        word_semantics[0][index_of_word[input_word]] = sentence_semantic
        learning_distribution[input_word][0] += 1 / dic[input_word]
        for output_word in output_words:
            in_train(input_word, output_word, w_in[0])

    else:
        if max(similarity_list)>=0.43:
            update_index = np.argmax(similarity_list)
            word_semantics[update_index][index_of_word[input_word]] += sentence_semantic
            learning_distribution[input_word][update_index] += 1 / dic[input_word]
            for output_word in output_words:
                in_train(input_word, output_word, w_in[update_index])
        else:
            if len(similarity_list)==meaning:
                update_index = np.argmax(similarity_list)
                learning_distribution[input_word][update_index] += 1 / dic[input_word]
                for output_word in output_words:
                    in_train(input_word, output_word, w_in[update_index])
            else:
                word_semantics[pivot][index_of_word[input_word]] = sentence_semantic
                update_index = pivot
                learning_distribution[input_word][update_index] += 1 / dic[input_word]
                for output_word in output_words:
                    in_train(input_word, output_word, w_in[update_index])



    step += 1

    if step % 100000 == 0:
        print('training!!! ', step * 100 / len(input), '% 완료')

print("training done!!!")


def save(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        word_index = index_of_word[word]
        for i in range(meaning):
            vector_str= ' '.join([str(s) for s in w_in[i][word_index]])
            f.write('%s %s %s\n' % (word, learning_distribution[word][i], vector_str))
    f.close()
    print("저장 완료!!!")


save('saved_matrix/test5')




















