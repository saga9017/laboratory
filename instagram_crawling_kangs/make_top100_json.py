import json

top100_dic=dict()

f=open("top100.txt", "r")
while True:
    line=f.readline()
    if not line:break
    keyword=line.split('\n')[0]
    print(keyword)
    top100_dic[keyword]=len(top100_dic)
f.close()

print(top100_dic)

with open('top100.json', 'w', encoding='utf-8') as make_file:
    json.dump(top100_dic, make_file)