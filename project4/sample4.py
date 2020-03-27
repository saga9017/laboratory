import os
import pickle
import random



origin_path="Faster-RCNN"
rcnns=os.listdir(origin_path)

post_dic={}
print(rcnns)

for i in rcnns:
    f = open(origin_path + '/' + i, 'r', encoding='utf-8')
    while True:
        line=f.readline()
        if not line:break
        print(line.split(' - '))
        post=line.split(' - ')[0].replace('.jpg', '').split('(')[0]
        rcnn=line.split(' - ')[1].replace('\n', '').replace('\t', ' ')
        if  post not in post_dic:
            post_dic[post]=rcnn
        else:
            post_dic[post] +=rcnn

    f.close()

# for i in post_dic:
#     print(i, post_dic[i])



origin_path="complete_crawling/img(0~98)"
hashtags=os.listdir(origin_path)
save = []

hashtag_dic={}
hashtag_count={}
vocab=[]
for i in hashtags:
    f = open(origin_path + '/' + i + '/' + i + '_preprocessed.txt', 'r', encoding='utf-8')
    f2 = open(origin_path+'/'+i+'/'+i+'_cap_preprocessed.txt', 'r', encoding='utf-8')
    f3 = open(origin_path+'/'+i+'/'+i+'_loc.txt', 'r', encoding='utf-8')
    line = f.readline()
    line2 = f2.readline()
    while True:
        tmp=[]
        line = f.readline()
        line2 = f2.readline()
        line3= f3.readline()
        if not line:break
        if (len(line.split('\t'))==2) & (len(line2.split('#')[1:])>0):
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
            tmp.append(line3.split('\t')[1].replace('\n', ''))
    f.close()
    f2.close()

score=0
new_hashtag_dic={}
for x, y in zip(hashtag_count, hashtag_count.values()):
    if y>15:
        score+=1
        new_hashtag_dic[x]=len(new_hashtag_dic)
print(score)
print(new_hashtag_dic)
print('vocab_size :', len(vocab))
print('==================================================================================')
for i in hashtags:
    f = open(origin_path + '/' + i + '/' + i + '_preprocessed.txt', 'r', encoding='utf-8')
    f2 = open(origin_path+'/'+i+'/'+i+'_cap_preprocessed.txt', 'r', encoding='utf-8')
    f3 = open(origin_path + '/' + i + '/' + i + '_loc.txt', 'r', encoding='utf-8')
    line = f.readline()
    line2 = f2.readline()
    while True:
        tmp=[]
        line=f.readline()
        line2=f2.readline()
        line3 = f3.readline()
        if not line:break

        if (len(line.split('\t'))==2) & (len(line2.split('#')[1:])>0):
            #print(line.split('\t')[1].replace('\n', ''))
            #print([k.split(' ')[0] for k in line2.split('#')[1:]])

            hashtag=[k.split(' ')[0] for k in line2.split('#')[1:]]
            tmp_hash=[]
            for j in hashtag:
                if hashtag_count[j]>15:
                    tmp_hash.append(new_hashtag_dic[j])
            if len(tmp_hash)>0:
                tmp.append(line.split('\t')[1].replace('\n', ''))
                tmp.append(tmp_hash)
                tmp.append(line3.split('\t')[1].replace('\n', ''))
                if line.split('\t')[0]!=line3.split('\t')[0]:
                    print('error1!!!')
                    continue
                else:
                    post_=i+'_'+line.split('\t')[0].replace('_', '')
                    if post_ not in post_dic:
                        print('error2!!!')
                        print(post_)
                        continue
                    else:
                        tmp.append(post_dic[post_])
                save.append(tmp)

    f.close()
    f2.close()

# for i in save:
#     print(i)
#
# print(len(save))
# random.shuffle(save)

# #save
# with open('data4.pickle', 'wb') as f:
#     pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)