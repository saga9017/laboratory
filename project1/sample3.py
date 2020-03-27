import os
import pickle

dic={}
f=open('project1_result2.txt')
for root, dirs, files in os.walk('NYT'):
    for fname in files:
        content=f.readline()
        print(fname, content)
        dic[fname]=content

f.close()


with open("transformer.pickle","wb") as fw:
    pickle.dump(dic, fw)
