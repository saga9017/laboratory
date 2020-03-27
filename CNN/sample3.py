from gensim.models import Word2Vec


input1=[]
input2=[]

f = open("rt-polaritydata/rt-polarity.neg", 'r', errors='ignore')
lines = f.readlines()
for line in lines:
    input1.append(line.replace('\n', '').replace(',', '').replace('.', '').split())
f.close()


f = open("rt-polaritydata/rt-polarity.pos", 'r', errors='ignore')
lines = f.readlines()
for line in lines:
    input2.append(line.replace('\n', '').replace(',', '').replace('.', '').split())
f.close()


input3=input1+input2


model = Word2Vec(input3[:100], size=300, window = 2, min_count=1, workers=4, iter=100, sg=1)


vocab = model.wv.vocab
index_of_word={x:y.index for x,y in vocab.items()}
X = model[vocab]

print(vocab['simplistic'])
print(index_of_word)
print(X)