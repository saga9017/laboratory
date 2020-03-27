import os
import shutil
import json
import pickle
import urllib.request



hashtagDic = {}
count = 0
f = open('./data/top100hashtags', 'r')
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
            for hash in hashtags:
                if hash in hashtagDic.keys():
                    hashtagDic[hash] += 1
                else:
                    hashtagDic[hash] = 1

            # images = lis[l]['img_urls']
            # for img in images:
            #     # imgName = 'img'+str(count)
            #     listt = lis[l]['key'].split('/')
            #     # print(listt[4])
            #     key = listt[4]
            #     # print(key)
            #     try:
            #         urllib.request.urlretrieve(img, './data/images/' + key + '.jpg')
            #         print(img)
            #         count += 1
            #     except:
            #         continue
            #     for hash in hashtags:
            #         if hash in hashtagDic.keys():
            #             hashtagDic[hash].append(key)
            #         else:
            #             hashtagDic[hash] = [key]
                # print(hashtagDic)
                        # urllib.request.urlretrieve(img, './data/img/'+hash+'/image'+str(hashtagDic[hash])+'.jpg')


    inputF.close()
f.close()
print('img file saved: ', count)
# print(len(hashtagDic))
# print(hashtagDic)
# count = 0
# for i in hashtagDic.keys():
#     dir_path = './data/img/'
#     dir_name = i
#     try:
#         os.mkdir(dir_path + dir_name + "/")
#     except FileExistsError:
#         continue
    # if hashtagDic[i] > 1:
    #     print(i)
    #     count += 1
# print(count)
# f = open('./data/tag2img.bin', 'wb')
# pickle.dump(hashtagDic, f)
# f.close()
# print(hashtagDic)
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
