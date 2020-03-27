import csv

max_len=1

label_dic={}
re_label_dic={}
label_num=0

y_train=[]
X_train=[]


with open('cnn_data/odp.wsdm.train.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break
        if int(line[0]) not in label_dic:
            label_dic[int(line[0])]=label_num
            re_label_dic[label_num]=int(line[0])
            label_num+=1

        y_train.append(label_dic[int(line[0])])
        X_train.append(' '.join(line[1:]))


print(X_train[:10])
print(y_train[:10])


y_dev=[]
X_dev=[]


with open('cnn_data/odp.wsdm.dev.all', 'r') as f:
    while True:
        line = f.readline().replace(',', '').split()
        if not line: break

        y_dev.append(label_dic[int(line[0])])
        X_dev.append(' '.join(line[1:]))


print(X_dev[:10])
print(y_dev[:10])


f=open('output_train.tsv', 'w', newline='')
csv_writer=csv.writer(f, delimiter='\t')
id=0
for i,j in zip(X_train, y_train):
    csv_writer.writerow([id, j, 0, i])
    id+=1
f.close()

f=open('output_dev.tsv', 'w', newline='')
csv_writer=csv.writer(f, delimiter='\t')
id=0
for i,j in zip(X_dev, y_dev):
    csv_writer.writerow([id, j, 0, i])
    id+=1
f.close()