import torch
import math
import random
from random import shuffle
from collections import Counter
import argparse
from huffman import HuffmanCoding

def Analogical_Reasoning_Task(embedding, w2i, i2w):
#######################  Input  #########################
# embedding : Word embedding (type:torch.tesnor(V,D))   #
#########################################################

    f = open("questions-words.txt", 'r')
    f_w = open("my_results_full_subsampling.txt", 'w')
    score=0
    #첫줄은 의미가 없으니 비우기 위함
    line = f.readline()
    while True:
        line = f.readline().split()
        if not line: break
        #학습되지 않은 단어일 경우 넘어감
        flag=True
        for i in line:
            if i not in w2i.keys():
                flag=False
        if flag==True:
            print('target :', line[3])
            calulated_vector=embedding[w2i[line[1]]]-embedding[w2i[line[0]]]+embedding[w2i[line[2]]]
            length = (embedding * embedding).sum(1) ** 0.5
            inputVector = calulated_vector.reshape(1,-1)
            sim = (inputVector @ embedding.t())[0] / length
            values, indices = sim.squeeze().topk(1)
            print('my result :', i2w[indices.item()])
            f_w.write(i2w[indices.item()]+'\n')
            if i2w[indices.item()]==line[3]:
                score += 1
        else:
            f_w.write('null' + '\n')
    f_w.close()
    f.close()
    print('score :', score)

def subsampling_table(freqdict):
###############################  Output  #########################################
# subsampled : Subsampled sequence                                               #
##################################################################################
    # subsampling을 위한 t
    t = 0.00001
    #word_seq는 frequency dictionary 형태로 받는다.
    total_word_number=sum(freqdict.values())
    dic_prob = {x: y / total_word_number for x, y in freqdict.items()}
    subsampling_prob = {x: 1 - math.sqrt(t / y) for x, y in dic_prob.items()}
    subsampled_dic={}
    for x, y in subsampling_prob.items():
        if y<0:
            subsampled_dic[x]=[1]
        else:
            subsampled_dic[x]=([0]*int(y*1000))+([1]*(1000-int(y*1000)))
    return subsampled_dic

def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################

    o=torch.mm(outputMatrix, inputMatrix[centerWord].view(-1,1))
    y = 1/(1+torch.exp(-1*o))

    j=0
    e = y.clone()
    z=0
    for i in contextCode:
        if i=='0':
            e[j]=e[j]
            z+=(-1*torch.log(1-y[j]))
        else:
            e[j]=e[j]-1
            z+=(-1*torch.log(y[j]))
        j+=1


###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    loss =torch.tensor(z)
    grad_in = torch.mm(outputMatrix.t(), e)
    grad_out = torch.mm(inputMatrix[centerWord].view(-1, 1), e.t()).t()

    return loss, grad_in, grad_out



def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    o=torch.mm(outputMatrix, inputMatrix[centerWord].view(-1,1))

    #y는 o에 softmax를 취해준 결과
    exp_o = torch.exp(o)
    sum_exp_o= torch.sum(exp_o)
    y = exp_o / sum_exp_o

    #e_j = y_j - t_j
    e = y.clone()
    e[-1]-=1

    # 내적을 위해 e를 2차원 배열로 만듦
    e = e.view(-1, 1)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    loss = -1 * torch.log(y)[-1]
    grad_in = torch.mm(outputMatrix.t(), e)
    grad_out = torch.mm(inputMatrix[centerWord].view(-1, 1), e.t()).t()

    return loss, grad_in, grad_out


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    o=torch.mm(outputMatrix, torch.sum(inputMatrix[contextWords], dim=0).view(-1,1))
    y = 1/(1+torch.exp(-1*o))

    j=0
    e = y.clone()
    z=0
    for i in centerCode:
        if i=='0':
            e[j]=e[j]
            z+=(-1*torch.log(1-y[j]))
        else:
            e[j]=e[j]-1
            z+=(-1*torch.log(y[j]))
        j+=1
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    loss = torch.tensor(z)
    grad_in = torch.mm(outputMatrix.t(), e).t()
    grad_out = torch.mm(torch.sum(inputMatrix[contextWords], dim=0).view(-1, 1), e.t()).t()

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    o=torch.mm(outputMatrix, torch.sum(inputMatrix[contextWords], dim=0).view(-1,1))

    #y는 o에 softmax를 취해준 결과
    exp_o = torch.exp(o)
    sum_exp_o= torch.sum(exp_o)
    y = exp_o / sum_exp_o

    #e_j = y_j - t_j
    e = y.clone()
    e[-1]-=1

    # 내적을 위해 e를 2차원 배열로 만듦
    e = e.view(-1, 1)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    loss = -1 * torch.log(y)[-1]
    grad_in = torch.mm(outputMatrix.t(), e).t()
    grad_out = torch.mm(torch.sum(inputMatrix[contextWords], dim=0).view(-1, 1), e.t()).t()

    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, full_tree, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=1):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    step=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    stats = torch.LongTensor(stats)

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            step+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    tree = full_tree
                    activated_nodes = [tree.index]
                    for i in codes[output]:
                        if i == '0':
                            tree = tree.left
                            activated_nodes.append(tree.index)
                        else:
                            tree = tree.right
                            activated_nodes.append(tree.index)
                    del activated_nodes[-1]
                    # print('activated_nodes :', activated_nodes)
                    activated = torch.tensor(activated_nodes)
                    L, G_in, G_out = CBOW_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_indices = torch.randint(0, len(stats), (NS,), dtype=torch.long)
                    activated = stats[random_indices]
                    # output과 겹치는 것이 있으면 다시 뽑는다.
                    while output in activated:
                        random_indices = torch.randint(0, len(stats), (NS,), dtype=torch.long)
                        activated = stats[random_indices]
                    activated = torch.cat((activated, torch.tensor([output])), 0)
                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    tree=full_tree
                    activated_nodes = [tree.index]
                    for i in codes[output]:
                        if i=='0':
                            tree=tree.left
                            activated_nodes.append(tree.index)
                        else:
                            tree=tree.right
                            activated_nodes.append(tree.index)
                    del activated_nodes[-1]
                    #print('activated_nodes :', activated_nodes)
                    activated=torch.tensor(activated_nodes)
                    L, G_in, G_out = skipgram_HS(inputs, codes[output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    random_indices = torch.randint(0, len(stats), (NS,), dtype=torch.long)
                    activated=stats[random_indices]
                    #output과 겹치는 것이 있으면 다시 뽑는다.
                    while output in activated:
                        random_indices = torch.randint(0, len(stats), (NS,), dtype=torch.long)
                        activated = stats[random_indices]
                    activated = torch.cat((activated, torch.tensor([output])), 0)
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out

                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L.item())
            if step%50000==0:
                avg_loss=sum(losses)/len(losses)
                print("Loss : %f" %(avg_loss,), end='   ')
                print((step/len(input_seq))*100, '% 완료')
                losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8.txt',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8.txt',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k


    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    codedict, full_tree= HuffmanCoding().build(freqdict)

    subsampled_dic=subsampling_table(freqdict)

    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])

    #Make training set
    print("build training set...")
    train_set = []
    input_set=[]
    target_set=[]
    window_size = 5
    if mode=="CBOW":
        for j in range(len(words)):
            #sampling_index=random.choice(subsampled_dic[w2i[words[j]]])
            sampling_index=1
            if sampling_index == 1:
                if j<window_size:
                    input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                    target_set.append(w2i[words[j]])
                elif j>=len(words)-window_size:
                    input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                    target_set.append(w2i[words[j]])
                else:
                    input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                    target_set.append(w2i[words[j]])
    if mode=="SG":
        for j in range(len(words)):
            #sampling_index=random.choice(subsampled_dic[w2i[words[j]]])
            sampling_index = 1
            if sampling_index == 1:
                if j<window_size:
                    input_set += [w2i[words[j]] for _ in range(window_size*2)]
                    target_set += [0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)]
                elif j>=len(words)-window_size:
                    input_set += [w2i[words[j]] for _ in range(window_size*2)]
                    target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)]
                else:
                    input_set += [w2i[words[j]] for _ in range(window_size*2)]
                    target_set += [w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()
    print(input_set[:100])
    #Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, full_tree ,freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01)

    Analogical_Reasoning_Task(emb, w2i, i2w)

    def save(file_name):
        f = open(file_name, 'w')
        for word in list(w2i.keys()):
            word_index = w2i[word]
            vector_str = ' '.join([str(s.item()) for s in emb[word_index]])
            f.write('%s %s\n' % (word, vector_str))

        f.close()
        print("저장 완료!!!")


    if mode=='SG':
        name='skip-gra'
    else:
        name='CBOW'

    if ns==0:
        name+='_hierarchical-softmax'
    else:
        name+='_negative-sampling'

    if part=='part':
        name+='_part'
    else:
        name+="_full"

    save(name+'_subsampling')

main()



