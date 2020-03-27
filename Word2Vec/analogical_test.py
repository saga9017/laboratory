import numpy as np
import torch


f = open("saved_matrix/text8_cbow_neg_vs.txt", 'r')

dic={}
embeding_matrix=[]
i=0
while True:
    line = f.readline().split(' ')
    if not line[0]: break
    dic[line[0]]=i
    word_matrix= [float(s) for s in line[1:] if s!='\n']
    embeding_matrix.append(word_matrix)
    i+=1

f.close()
reverse_dic = {y:x for x,y in dic.items()}

print(np.array(list(dic.keys())).shape)
embedding=torch.tensor(embeding_matrix)
print(embedding.shape)



total=0
f = open("questions-words.txt", 'r')
f_w = open("my_results_full_test.txt", 'w')
score=0
#첫줄은 의미가 없으니 비우기 위함
line = f.readline()
while True:
    total+=1
    line = f.readline().split()
    if not line: break
    #학습되지 않은 단어일 경우 넘어감
    flag=True
    for i in line:
        if i not in dic.keys():
            flag=False
    if flag==True:
        #print('target :', line[3])
        calulated_vector=embedding[dic[line[1]]]-embedding[dic[line[0]]]+embedding[dic[line[2]]]
        length = (embedding * embedding).sum(1) ** 0.5
        inputVector = calulated_vector.reshape(1,-1)
        sim = (inputVector @ embedding.t())[0] / length
        values, indices = sim.squeeze().topk(4)
        for j in indices:
            if j not in [dic[line[0]], dic[line[1]], dic[line[2]]]:
                right_index=j.item()
                break
        #print('my result :', reverse_dic[indices.item()])
        f_w.write(reverse_dic[right_index]+'\n')
        if reverse_dic[right_index]==line[3]:
            score += 1
    else:
        f_w.write('null' + '\n')
f_w.close()
f.close()
print('score :', score/total)
print(total)