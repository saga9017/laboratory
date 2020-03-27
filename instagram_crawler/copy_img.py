import shutil
import pickle

f = open('./data/tag2img.bin', 'rb')
tag2img = pickle.load(f)
f.close()

f = open('./data/tags.txt', 'w', encoding='utf-8')
for tag in tag2img.keys():
    # for key in tag2img[tag]:
    #     shutil.copy('./data/images/'+key+'.jpg', '../multimodal-deep-learning-for-disaster-response-master/data/img/'+tag+'/')
    print(tag)
    f.write(tag+'\n')

print('img saved done')
f.close()