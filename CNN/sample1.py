

input1=[]
input2=[]

f = open("rt-polaritydata/rt-polarity.neg", 'r', errors='ignore')
lines = f.readlines()
for line in lines:
    input1.append(line.replace('\n', '').replace(',', '').replace('.', ''))
f.close()

train_x=input1[:int((len(input1)*9)/10)]
test_x=input1[int((len(input1)*9)/10):]

train_y=[0]*len(train_x)
test_y=[0]*len(test_x)


f = open("rt-polaritydata/rt-polarity.pos", 'r', errors='ignore')
lines = f.readlines()
for line in lines:
    input2.append(line.replace('\n', '').replace(',', '').replace('.', ''))
f.close()

train_x.extend(input2[:int((len(input2)*9)/10)])
test_x.extend(input2[int((len(input1)*9)/10):])

train_y.extend([1]*len(input2[:int((len(input2)*9)/10)]))
test_y.extend(([1]*len(input2[int((len(input1)*9)/10):])))

