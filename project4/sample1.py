import os
import pickle

origin_path="complete_crawling/img(0~98)"
hashtags=os.listdir(origin_path)
save = []

hashtag_dic={}
hashtag_count={}
vocab=[]
for i in hashtags:
    f = open(origin_path + '/' + i + '/' + i + '_preprocessed.txt', 'r', encoding='utf-8')
    f2 = open(origin_path+'/'+i+'/'+i+'_cap_preprocessed.txt', 'r', encoding='utf-8')
    while True:
        tmp=[]
        line=f.readline()
        line2=f2.readline()
        if not line:break
        if len(line.split('\t'))==2 & len(line2.split('#')[1:])>0:
            #print(line.split('\t')[1].replace('\n', ''))
            #print([k.split(' ')[0] for k in line2.split('#')[1:]])
            sentence=line.split('\t')[1].replace('\n', '')
            # for j in sentence.split():
            #     if j not in vocab:
            #         vocab.append(j)
            tmp.append(sentence)
            hashtag=[k.split(' ')[0] for k in line2.split('#')[1:]]
            tmp_hash=[]
            for j in hashtag:
                if j not in hashtag_dic:
                    hashtag_dic[j]=len(hashtag_dic)
                    hashtag_count[j]=1
                    tmp_hash.append(hashtag_dic[j])
                else:
                    hashtag_count[j]+=1
                    tmp_hash.append(hashtag_dic[j])
            tmp.append(tmp_hash)
    f.close()
    f2.close()

score=0
new_hashtag_dic={}
for x, y in zip(hashtag_count, hashtag_count.values()):
    if y>9:
        score+=1
        new_hashtag_dic[x]=len(new_hashtag_dic)
print(score)
print(new_hashtag_dic)
print('vocab_size :', len(vocab))

for i in hashtags:
    f = open(origin_path + '/' + i + '/' + i + '_preprocessed.txt', 'r', encoding='utf-8')
    f2 = open(origin_path+'/'+i+'/'+i+'_cap_preprocessed.txt', 'r', encoding='utf-8')
    while True:
        tmp=[]
        line=f.readline()
        line2=f2.readline()
        if not line:break
        if len(line.split('\t'))==2 & len(line2.split('#')[1:])>0:
            #print(line.split('\t')[1].replace('\n', ''))
            #print([k.split(' ')[0] for k in line2.split('#')[1:]])

            hashtag=[k.split(' ')[0] for k in line2.split('#')[1:]]
            tmp_hash=[]
            for j in hashtag:
                if hashtag_count[j]>9:
                    tmp_hash.append(new_hashtag_dic[j])
            if len(tmp_hash)>0:
                tmp.append(line.split('\t')[1].replace('\n', ''))
                tmp.append(tmp_hash)
                save.append(tmp)
    f.close()
    f2.close()

for i in save:
    print(i)

print(len(save))

# #save
# with open('data.pickle', 'wb') as f:
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)