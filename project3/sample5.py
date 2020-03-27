import random


f1=open('fine-tune_dataset.txt', 'r', encoding='utf-8')
f2=open('fine-tune_label.txt', 'r', encoding='utf-8')

dataset=[]
while True:
    line1=f1.readline()
    line2=f2.readline()
    if not line1: break
    temp=[line1, line2]
    dataset.append(temp)


f1.close()
f2.close()


random.shuffle(dataset)
print(dataset)

f_w=open('train.txt', 'w', encoding='utf-8')
f_w2=open('test.txt', 'w', encoding='utf-8')

for x, y in dataset[:15000]:
    f_w.write(x.replace('\n', '').replace('\t', ' ')+'\t'+y.replace('\n', ''))
    f_w.write('\n')

for x, y in dataset[15000:]:
    f_w2.write(x.replace('\n', '').replace('\t', ' ')+'\t'+y.replace('\n', ''))
    f_w2.write('\n')

f_w.close()
f_w2.close()