from close_word import similar_words_dic
import load_cooccurence_matrix
# skip-gram_hierarchical-softmax
import numpy as np
import random
import math
import time
import operator

# 파일을 열어 단어를 읽어 온다.
f = open("text9.txt", 'r')
words = f.read().split(' ')
f.close()

# text에 있는 전체 단어의 수를 센다.
total_word_number = np.array(words).shape[0]
print('단어 수 :', total_word_number)

# first_dic에는 (word: 그 단어수)의 dictionary를 부여한다.
prime_dic = {}

# dic에는 unknown이라는 단어를 포함한다.
dic = {}

# index_of_word에는 해당 단어의 index, 배열 내 위치 를 저장한다.
index_of_word = {}

embeding_size = 100
window_size = 2

# learning rate
alpha = 0.025

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

vocabulary_size = np.array(list(index_of_word.keys())).shape[0]
print('vocabulary_size :', vocabulary_size)
#print(index_of_word)

dic_prob = {x: y / total_word_number for x, y in dic.items()}
print('dic_prob :', dic_prob)


# input, output 만들기
for i in range(len(words)):
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

"""""
for i in range(32):
    print(input[i],'->',output[i])
"""""


# 각 단어의 고유번호를 onehot vector로 만들기위한 함수
def one_hot_encoding(word):
    one_hot_vector = [0] * (vocabulary_size)
    index = index_of_word[word]
    one_hot_vector[index] = 1
    return one_hot_vector


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y





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
        self.data = data
        self.index=None
        self.prob=None
        self.parent=None
        self.left = self.right=None



#huffman code의 생성
huffman_code=dic_prob.copy()

nodes={}
node_index=-1


while len(huffman_code.keys())!=1:
    a1, a2 = lowest_prob_pair(huffman_code)
    huffman_code[(a1[0], a2[0])] = a1[1] + a2[1]
    del huffman_code[a1[0]]
    del huffman_code[a2[0]]
    nodes[(a1[0], a2[0])]=Node((a1[0], a2[0]))
    node=nodes[(a1[0], a2[0])]
    node_index += 1
    node.index = node_index
    node.prob=a1[1]/(a1[1]+a2[1])

    if a1[0] in nodes.keys():
        node.left=nodes[a1[0]]
        nodes[a1[0]].parent=node
    else:
        nodes[a1[0]]=Node(a1[0])
        node.left = nodes[a1[0]]
        nodes[a1[0]].parent = node

    if a2[0] in nodes.keys():
        node.right = nodes[a2[0]]
        nodes[a2[0]].parent = node
    else:
        nodes[a2[0]] = Node(a2[0])
        node.right = nodes[a2[0]]
        nodes[a2[0]].parent = node



root=list(huffman_code.keys())[0]
root_node=nodes[root]
huffman_code[root]=''



while len(huffman_code.keys())<len(dic_prob.keys()):
    keys=list(huffman_code.keys())

    for i in range(len(keys)):
        if keys[0] in dic_prob.keys():
            del keys[0]


    huffman_code[keys[0][0]]=huffman_code[keys[0]]+'0'
    huffman_code[keys[0][1]]=huffman_code[keys[0]]+'1'

    del huffman_code[keys[0]]



print('huffman_code :', huffman_code)

w_in = np.array([[random.uniform(-1, 1) for i in range(embeding_size)] for j in range(vocabulary_size)])
w_out = np.array([[random.uniform(-1, 1) for i in range(embeding_size)] for j in range(node_index+1)]).T
print('w_in.shape :', w_in.shape)
print('w_out.shape :', w_out.shape)



index = 0
# train하는 과정
for i in range(len(input)):
    # input, output 단어
    if input[i] not in dic.keys():
        input_word = 'UNKNOWN'
    else:
        input_word = input[i]

    if output[i] not in dic.keys():
        output_word = 'UNKNOWN'
    else:
        output_word = output[i]

    # 단어의 index
    input_word_number = index_of_word[input_word]
    output_word_number = index_of_word[output_word]

    #output단어의 huffman code
    output_huffman_code=huffman_code[output_word]
    active_node=[]
    parent(nodes[output_word], active_node)
    active_node.reverse()

    hierarchi_samples = np.array([w_out.T[i] for i in active_node])

    y = sigmoid(np.dot(hierarchi_samples, w_in[input_word_number]))

    j=0
    e=np.zeros(y.shape)
    for i in output_huffman_code:
        if i=='0':
            e[j]=y[j]
        else:
            e[j]=y[j]-1
        j+=1


    # 내적을 위해 e를 2차원 배열로 만듦
    e = np.reshape(e, (-1, 1))


    temp = np.reshape(np.dot(hierarchi_samples.T, e), -1)


    # w_out을 먼저 update 한다.
    j = 0
    for i in active_node:
        w_out.T[i] -= alpha * w_in[input_word_number] * e[j]
        j += 1

    # w_out을 update한 뒤 w_in을 update 한다.
    w_in[input_word_number] -= alpha * temp

    index += 1

    if index % 10000 == 0:
        print('training!!!')

print("training done!!!")


def save(file_name):
    f = open(file_name, 'w')
    for word in list(dic.keys()):
        word_index = index_of_word[word]
        vector_str = ' '.join([str(s) for s in w_in[word_index]])
        f.write('%s %s\n' % (word, vector_str))

    f.close()
    print("저장 완료!!!")


save('adagram')

