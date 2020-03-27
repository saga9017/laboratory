f=open('[lstm_attention]view_sentence=6669 batch_size=2048 epoch=50.txt', 'r')

lines=f.readlines()

f.close()


input=lines[2::4]
result=lines[3::4]
predict=lines[0::4]
del predict[0]

table_result={}

for i, j in zip(input, result):
    if i[8:-1] not in table_result:
        table_result[i[8:-1]]=[j[9:-1]]
    else:
        table_result[i[8:-1]].append(j[9:-1])


table_predict={}
for i, j  in zip(input, predict):
    if i[8:-1] not in table_predict:
        table_predict[i[8:-1]]=[' '.join(j.split('EOS')[0].replace('SOS', '').replace('predict', '').replace(':', '').replace('[', '').replace(']', '').replace("'", '').replace(',', '').split())]
    else:
        table_predict[i[8:-1]].append(' '.join(j.split('EOS')[0].replace('SOS', '').replace('predict', '').replace(':', '').replace('[', '').replace(']', '').replace("'", '').replace(',', '').split()))

#print(table_result)
#print(table_predict)

#candidate=[]
total_score=0
total_word=0
for x,y in zip(table_result.values(), table_predict.values()):
    score = 0
    for i in range(len(x)):
        if x[i]==y[i]:
            score=1
    if score==0:
        print(x, y)
        """""
        for k,v  in table_result.items():
            if v==x:
                candidate.append(k)
        """""
    total_score+=score
    total_word+=1

print(total_word, '중에 일치 문장 :', total_score)

"""
f=open('candidate.txt', 'w')
for i in candidate:
    f.write(i)
    f.write('\n')
f.close()
"""