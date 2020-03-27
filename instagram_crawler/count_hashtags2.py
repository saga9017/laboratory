import os
import shutil
import json
import pickle
import urllib.request

"""""
##dictionary 변환 과정
alldic = dict()
tag2key = dict()
key2info = dict()
imgcount = 0
for root, dirs, files in os.walk('./data/hashtag'):
    for fname in files:
        # print(fname)
        dire = './data/hashtag/'+fname
        f = open(dire, 'r', encoding='utf-8')
        while True:
            line = f.readlines()
            if not line:
                break
            dict = json.loads(line[0])
            for li in dict:
                key = li['key'].split('/')[4]
                alldic[key] = li
                imgcount += len(alldic[key]['img_urls'])
                # print(li['hashtags'])
                for tag in li['hashtags']:
                    if tag in tag2key.keys():
                        if key not in tag2key[tag]:
                            tag2key[tag].append(key)
                    else:
                        tag2key[tag] = [key]
print(len(tag2key))


f = open('./data/tag2key.bin', 'wb')
pickle.dump(tag2key, f)
f.close()

f = open('./data/key2info.bin', 'wb')
pickle.dump(alldic, f)
f.close()

print('imgcount=', imgcount)
print('post count=', len(alldic))
print('tag=', len(tag2key))


###### GERNERATE TAG2IMG ########
f = open('./data/tag2key.bin', 'rb')
tag2key = pickle.load(f)
f.close()


f = open('./data/key2info.bin', 'rb')
key2info = pickle.load(f)
print(key2info)
f.close()

tag2img = dict()
count = 0
for tag in tag2key.keys():
    print(tag)
    tag2img[tag] = list()
    for key in tag2key[tag]:
        #print(key, key2info[key]['img_urls'])
        for img in key2info[key]['img_urls']:
            img_url = img.split('/')
            img_name=img_url[-1].split('.')[0]
            # if len(img_url) == 10:
            #     img_name = img_url[9].split('.')[0]
            # elif len(img_url) == 9:
            #     img_name = img_url[8].split('.')[0]
            # else:
            #     img_name = img_url[10].split('.')[0]
            tag2img[tag].append(img_name)

f = open('./data/tag2img.bin', 'wb')
pickle.dump(tag2img, f)
f.close()

print(tag2img)

print(count)

f = open('./data/key2info.bin', 'rb')
key2info = pickle.load(f)
print(key2info)
f.close()

count = 0
for key in key2info.keys():
    print(key)
    for img in key2info[key]['img_urls']:
        img_url = img.split('/')
        img_name = img_url[-1].split('.')[0]
        # if len(img_url) == 10:
        #     img_name = img_url[9].split('.')[0]
        # elif len(img_url) == 9:
        #     img_name = img_url[8].split('.')[0]
        # else:
        #     img_name = img_url[10].split('.')[0]
        try:
            urllib.request.urlretrieve(img, './data/images/' + img_name + '.jpg')
            count += 1
        except:
            continue

print(count)

"""""


f = open('./data/key2info.bin', 'rb')
key2info = pickle.load(f)
print(key2info)
f.close()

count = 0
key2text = dict()
for key in key2info.keys():
    print(key)
    # print(key2info[key]['caption'])
    # print("================================================")
    caption = key2info[key]['caption']
    captionList = caption.split(' ')
    # print(captionList)
    newCaption = list()
    for word in captionList:
        if '#' in word:
            # print(word)
            continue
        else:
            newCaption.append(word)
    # print((' '.join(newCaption)))
    # print(key2info[key]['hashtags'])
    th = dict()
    th['text'] = ' '.join(newCaption)
    th['hashtags'] = key2info[key]['hashtags']
    key2text[key] = th
    count += 1
print(count)


f = open('./data/key2text.bin', 'wb')
pickle.dump(key2text, f)
f.close()


f = open('./data/tag2img.bin', 'rb')
tag2img = pickle.load(f)
f.close()


count = 0
for tag in tag2img.keys():
    print(tag, len(tag2img[tag]))
    if len(tag2img[tag]) < 4:
        count += 1
print(count)
print(len(tag2img))
print(len(tag2img)-count)

hashtagDic = {}
count = 0
f = open('./data/100hashtagNew', 'r', encoding='utf8')
while True:
    line = f.readline()
    if not line:
        break
    tag = line[1:len(line) - 1]
    # print(tag)
    outputFile = "./data/hashtag/" + tag

    inputF = open(outputFile, 'rt', encoding='UTF8')
    lines = inputF.readlines()
    for line in lines:
        # print(line)
        lis = json.loads(line)
        for l in range(100):
            hashtags = lis[l]['hashtags']
            count += len(hashtags)
            # for hash in hashtags:
            #     if hash in hashtagDic.keys():
            #         hashtagDic[hash] += 1
            #     else:
            #         hashtagDic[hash] = 1

            images = lis[l]['img_urls']
            for img in images:
                # imgName = 'img'+str(count)
                listt = lis[l]['key'].split('/')
                # print(listt[4])
                key = listt[4]
                # print(key)
                try:
                    urllib.request.urlretrieve(img, './data/images/' + key + '.jpg')
                    print(img)
                    count += 1
                except:
                    continue
                for hash in hashtags:
                    if hash in hashtagDic.keys():
                        hashtagDic[hash].append(key)
                    else:
                        hashtagDic[hash] = [key]
                    #print(hashtagDic)
                    if os.path.isdir("./data/img/"+hash):
                        urllib.request.urlretrieve(img, './data/img/'+hash+'/image'+str(hashtagDic[hash])+'.jpg')
                    else:
                        os.makedirs("./data/img/"+hash)
                        urllib.request.urlretrieve(img, './data/img/' + hash + '/image' + str(hashtagDic[hash]) + '.jpg')

    inputF.close()
f.close()
print('img file saved: ', count)
print(len(hashtagDic))
print(hashtagDic)
count = 0




import numpy as np
f = open('./data/key2info.bin', 'rb')
key2info = pickle.load(f)
f.close()

text = list()
tags = list()

for i in key2info.keys():
    text.append(key2info[i]['caption'])
    tags.append(key2info[i]['hashtags'])

print(len(tags))
print(len(text))
print(np.array(text).shape)
text = np.append(np.array(text), np.array(tags), axis=0)
# text.append(tags)
# print(text)
print(np.array(text).shape)
text.dump(open('./data/itag.npz', 'wb'))


f = open('./data/itag.npz', 'wb')
text = np.array(text)
pickle.dump(text, f)
f.close()


f = open('./data/tag2key.bin', 'rb')
tag2key = pickle.load(f)
f.close()

count = 0
tcount = 0
for ks in tag2key.values():
    # print(len(ks))
    if len(ks) > 10:
        count += len(ks)
        tcount += 1
print('aver=', count/tcount)
print(tcount)
print('imgcount=', count)



for i in hashtagDic.keys():
    dir_path = './data/img/'
    dir_name = i
    try:
        os.mkdir(dir_path + dir_name + "/")
    except FileExistsError:
        continue
    if hashtagDic[i] > 1:
        print(i)
        count += 1
print(count)
f = open('./data/tag2img_big.bin', 'wb')
pickle.dump(hashtagDic, f)
f.close()
print(hashtagDic)


print(count)
print(count/10000)
print(len(hashtagDic))
f = open('./data/tag2img.bin', 'rb')
tag2img = pickle.load(f)
print(len(tag2img))
c = 0
max = 0
for t, tag in tag2img.items():
    c += len(tag)
    if max < len(tag):
        max = len(tag)

        mtag = t
print(c/len(tag2img))
print(max)
print(mtag)
