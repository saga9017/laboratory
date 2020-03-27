import math
import nltk.translate.bleu_score as bleu
import nltk
print(nltk.__version__)

f = open('[basic_lstm]view_sentence=6669 batch_size=2048 epoch=50.txt', 'r')

lines = f.readlines()

f.close()

input = lines[2::4]
result = lines[3::4]
predict = lines[0::4]
del predict[0]

table_result = {}

for i, j in zip(input, result):
    if i[8:-1] not in table_result:
        table_result[i[8:-1]] = [j[9:-1]]
    else:
        table_result[i[8:-1]].append(j[9:-1])

# print(table)

table_predict = {}
for i, j in zip(input, predict):
    if i[8:-1] not in table_predict:
        table_predict[i[8:-1]] = [' '.join(
            j.split('EOS')[0].replace('SOS', '').replace('predict', '').replace(':', '').replace('[', '').replace(']',
                                                                                                                  '').replace(
                "'", '').replace(',', '').split())]
    else:
        table_predict[i[8:-1]].append(' '.join(
            j.split('EOS')[0].replace('SOS', '').replace('predict', '').replace(':', '').replace('[', '').replace(']',
                                                                                                                  '').replace(
                "'", '').replace(',', '').split()))

# print(table_result)
# print(table_predict)

x = ['1', '2', '3', '4', '5', '6', '7', '8']


def make_ngram(x):
    dic = {}
    for i in range(4):
        temp = []
        for j in range(len(x) - i):
            temp.append('_'.join(x[j:j + i + 1]))
        dic[i + 1] = ' '.join(temp)
    return dic


print(make_ngram(x))

"""""
score=0
total_word=0
final_score=[]
for x,y in zip(table_result.values(), table_predict.values()):
    score=[]
    temp=[]
    for result_s, predict_s in zip(x,y):
        sentence_score=1

        result_ngram_dic=make_ngram(result_s.split())
        predict_ngram_dic=make_ngram(predict_s.replace('EOS', '').split())
        for z in range(4):
            result_word_count = {}
            for i in result_ngram_dic[z+1].split():
                if i not in result_word_count:
                    result_word_count[i]=1
                else:
                    result_word_count[i]+=1

            matched_word_count={}
            for i in predict_ngram_dic[z+1].split():
                if i in result_ngram_dic[z+1].split():
                    if i not in matched_word_count:
                        matched_word_count[i]=1
                    else:
                        matched_word_count[i] += 1

            for i,j in matched_word_count.items():
                if j>result_word_count[i]:
                    matched_word_count[i]=result_word_count[i]

            print(result_ngram_dic[z+1].split())
            print(predict_ngram_dic[z+1].split())
            print(result_word_count)
            print(matched_word_count)

            if len(predict_ngram_dic[z+1])==0 or len(result_ngram_dic[z+1])==0:
                sentence_score*=1
            else:
                sentence_score *= sum(matched_word_count.values())/len(predict_ngram_dic[z+1].split())

            if len(predict_ngram_dic[z+1].split())==0:
                beta=1
            else:
                beta=math.pow(math.e, min(0, 1-len(result_ngram_dic[z+1].split())/len(predict_ngram_dic[z+1].split())))

        score.append(beta*math.pow(sentence_score, 4))
    final_score.append(max(score))

print(final_score)
print('BLEU :', sum(final_score)/len(final_score)*100)
"""""

a = 'm a teacher here .'
b = 'meet thing is afraid .'
b = 'm a teacher here .'
print(a.split())
print(b.split())
print(bleu.sentence_bleu([a.split()], b.split()))
#print(bleu.sentence_bleu([a], [b]))

total_score = []
for x, y in zip(table_result.values(), table_predict.values()):
    max_score = 0
    for i, j in zip(x, y):
        if len(i.split()) >= 4:
            weights = (0.25, 0.25, 0.25, 0.25)
            score = bleu.sentence_bleu([j.split()], i.split(), weights)
        else:
            weights = (1 / len(i.split()),) * len(i.split())
            score = bleu.sentence_bleu([j.split()], i.split(), weights)
        if max_score < score:
            max_score = score

    total_score.append(max_score)

print('sentence bleu :', sum(total_score) / len(total_score))


total_score = []
for x, y in zip(table_result.values(), table_predict.values()):
    max_score = 0
    reference=[]
    for i in x:
        reference.append(i.split())
    hypothesis = y[0].split()

    total_score.append(bleu.sentence_bleu(reference, hypothesis))


print('sentence bleu :', sum(total_score) / len(total_score))




list_of_reference=[]
hypothesis=[]
for x, y in zip(table_result.values(), table_predict.values()):
    temp=[]
    for i in x:
        temp.append(i.split())
    list_of_reference.append(temp)
    hypothesis.append(y[0].split())


print('corpus bleu :', bleu.corpus_bleu(list_of_reference, hypothesis))

