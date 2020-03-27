import os
import re
import random
import numpy as np

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


os.chdir(r'C:/Users/Lee Wook Jin/PycharmProjects/Fasttext/A. Multi-Class')

# AG Data hyperparamters
AG = dict()
AG["Max_length(Unigram)"] = 50
AG["Max_length(Bigram)"] = 100
AG["Path"] = "ag_news_csv"
AG["Class"] = 4
AG["Learning_rate"] = 0.0005
AG["Batch"] = 256

# Sogou Data hyperparamters
Sogou = dict()
Sogou["Max_length(Unigram)"] = 1000
Sogou["Max_length(Bigram)"] = 1500
Sogou["Path"] = "sogou_news_csv"
Sogou["Class"] = 5
Sogou["Learning_rate"] = 0.0003
Sogou["Batch"] = 1024

# Amazon Full Data hyperparamters
Amz_F = dict()
Amz_F["Max_length(Unigram)"] = 100
Amz_F["Max_length(Bigram)"] = 500
Amz_F["Path"] = "amazon_review_full_csv"
Amz_F["Class"] = 5
Amz_F["Learning_rate"] = 0.0001
Amz_F["Batch"] = 1024

# Amazon Polarity Data hyperparamters
Amz_P = dict()
Amz_P["Max_length(Unigram)"] = 100
Amz_P["Max_length(Bigram)"] = 350
Amz_P["Path"] = "amazon_review_polarity_csv"
Amz_P["Class"] = 2
Amz_P["Learning_rate"] = 0.0001
Amz_P["Batch"] = 1024

# DBPedia Data hyperparamters
DBP = dict()
DBP["Max_length(Unigram)"] = 50
DBP["Max_length(Bigram)"] = 100
DBP["Path"] = "dbpedia_csv"
DBP["Class"] = 14
DBP["Learning_rate"] = 0.0005
DBP["Batch"] = 256

# Yahoo answer Data hyperparamters
Yah_A = dict()
Yah_A["Max_length(Unigram)"] = 150
Yah_A["Max_length(Bigram)"] = 300
Yah_A["Path"] = "yahoo_answers_csv"
Yah_A["Class"] = 10
Yah_A["Learning_rate"] = 0.0001
Yah_A["Batch"] = 1024

# Yelp Polarity Data hyperparamters
Yelp_P = dict()
Yelp_P["Max_length(Unigram)"] = 250
Yelp_P["Max_length(Bigram)"] = 500
Yelp_P["Path"] = "yelp_review_polarity_csv"
Yelp_P["Class"] = 2
Yelp_P["Learning_rate"] = 0.00025
Yelp_P["Batch"] = 1024

# Yelp Full Data hyperparamters
Yelp_F = dict()
Yelp_F["Max_length(Unigram)"] = 250
Yelp_F["Max_length(Bigram)"] = 500
Yelp_F["Path"] = "yelp_review_full_csv"
Yelp_F["Class"] = 5
Yelp_F["Learning_rate"] = 0.00025
Yelp_F["Batch"] = 1024

DataName = dict()
DataName["AG"] = AG
DataName["Sogou"] = Sogou
DataName["DBP"] = DBP
DataName["Yelp P."] = Yelp_P
DataName["Yelp F."] = Yelp_F
DataName["Yah. A."] = Yah_A
DataName["Amz F."] = Amz_F
DataName["Amz P."] = Amz_P


def RawData(PATH):
    # Loading train and test data
    # return [sentence,label] (not split sentence) (because of efficient memory)

    # Loading train data
    print("The train data is being loaded")
    train_open = open(PATH + "/train.csv", 'r', encoding='utf-8')
    train = [[clean_str(lines)[4:], int(clean_str(lines)[0])] for lines in
             train_open.readlines()]  # clean_str : data prerpreocessing

    # Loading test data
    print("The test data is being loaded")
    test_open = open(PATH + "/test.csv", 'r', encoding='utf-8')
    test = [[clean_str(lines)[4:], int(clean_str(lines)[0])] for lines in
            test_open.readlines()]  # clean_str : data prerpreocessing

    random.shuffle(train)

    return train, test


def Dictionary(train, test, Bigrams=False):
    # Making Bigram Function
    def Bigram(x):
        edit = []
        for index in range(len(x) - 1):
            edit.append('%s_%s' % (x[index], x[index + 1]))
        return edit

    word_dict = dict()
    word_dict["#PAD"] = 0  # For padding (mini-batch)

    # Unigram
    if Bigrams == False:
        print("< Unigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split()  # train sentece split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word] = len(word_dict)  # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split()  # test sentence split

            for word in sentence:
                if not word in word_dict:
                    word_dict[word] = len(word_dict)  # making dictionary

        return word_dict

    else:
        print("< Unigram + Bigram > Word dictionary is being made")
        for lines in train:
            sentence = lines[0].split()  # train sentece split
            sentence += Bigram(sentence)  # Adding Bigram

            for word in sentence:
                if not word in word_dict:
                    word_dict[word] = len(word_dict)  # making dictionary

        # Adding test words
        for lines in test:
            sentence = lines[0].split()  # test sentence split
            sentence += Bigram(sentence)  # Adding Bigram

            for word in sentence:
                if not word in word_dict:
                    word_dict[word] = len(word_dict)  # making dictionary

        return word_dict


def Padding(data, length, word_dict, Bigrams=False):
    # Making Bigram Function
    def Bigram(x):
        edit = []
        for index in range(len(x) - 1):
            edit.append('%s_%s' % (x[index], x[index + 1]))
        return edit

    sentence_index = []

    if Bigrams == False:
        print("< Unigram > The data spliting is being loaded")

        # Unigram
        for lines in data:
            sentence = lines[0].split()  # sentence split
            label = lines[1] - 1  # because of difference between label and index

            edit = []
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit, label])

        return sentence_index

    else:
        print("< Unigram + Bigram > The data spliting is being loaded")

        # Unigram + Bigram
        for lines in data:
            sentence = lines[0].split()  # sentence split
            sentence += Bigram(sentence)  # Adding Bigram
            label = lines[1] - 1  # because of difference between label and index

            edit = []
            for word in sentence:
                edit.append(word_dict[word])

            sentence_index.append([edit, label])

        return sentence_index


def Spliting(train, test, valid_rate):
    print("Split train - dev")
    valid_rate = 0.05

    random.shuffle(train)  # For making data sequences randomly
    train_ = train[:int(len(train) * (1 - valid_rate))]  # train set
    valid_ = train[int(len(train) * (1 - valid_rate)):]  # validation set
    test_ = test  # test set

    return train_, valid_, test_

name='Yah. A.'
bigram=False
Route=DataName[name]
if bigram==False:
    print("fastText for Text Classification")
    train_raw, test_raw = RawData(Route["Path"])
    word_dict=Dictionary(train_raw, test_raw,Bigrams=False)
    train_split = Padding(train_raw,Route["Max_length(Unigram)"],word_dict)
    test_split = Padding(test_raw,Route["Max_length(Unigram)"],word_dict)
    train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)

else:
    print("fastText for Text Classification")
    train_raw, test_raw = RawData(Route["Path"])
    word_dict=Dictionary(train_raw, test_raw,True)
    train_split = Padding(train_raw,Route["Max_length(Bigram)"],word_dict,True)
    test_split = Padding(test_raw,Route["Max_length(Bigram)"],word_dict,True)
    train,valid,test = Spliting(train_split,test_split,valid_rate=0.05)



##################################################################################################


train_np=np.array(train)
x=train_np[:,:-1]
label=train_np[:,-1]
embeding_size=10
output_size=10
alpha=0.01
epoch=5
l2_norm=0.0025



#index_of_word에는 해당 단어의 index, 배열 내 위치 를 저장한다.
index_of_word=word_dict

print(list(index_of_word)[:10])

vocabulary_size=len(index_of_word)
print('vocabulary_size :', vocabulary_size)


w_in=np.random.uniform(-0.01,0.01,(vocabulary_size,  embeding_size))
w_out=np.random.uniform(-0.01,0.01,(output_size, embeding_size)).T

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

print('training 개수 :', len(label))
step=0
loss_list=[]
for ep in range(epoch):
    print('epoch :', ep)
    for i in range(len(label)):
        # input, output 단어

        # 단어의 index
        input_word_numbers =x[i][0]
        output_label = int(label[i])

        """""
        w_ins=w_in[input_word_numbers]
        norm_w_ins=(w_ins-np.mean(w_ins, axis=1).reshape(-1,1))/np.std(w_ins, axis=1).reshape(-1,1)
        sum_w_in = np.sum(norm_w_ins, axis=0)
        """
        sum_w_in=np.sum(w_in[input_word_numbers], axis=0)
        y = softmax(np.dot(w_out.T, sum_w_in))

        loss=-np.log(y[output_label])

        #y와 결과값을 계산
        e = np.copy(y)
        e[output_label]-=1

        # 내적을 위해 e를 2차원 배열로 만듦
        e = np.reshape(e, (-1, 1))


        temp=np.reshape(np.dot(w_out,e), -1)



        #w_out을 먼저 update 한다.
        w_out-= alpha * np.dot(sum_w_in.reshape(-1,1) , e.T)+l2_norm*w_out


        #w_out을 update한 뒤 w_in을 update 한다.
        for i in input_word_numbers:
            w_in[i] -= alpha *temp+l2_norm*w_in[i]


        loss_list.append(loss)
        step += 1
        if step % 10000 == 0:
            print('training!!!   ', end='')
            print('loss :', np.sum(loss_list)/10000)
            loss_list=[]

    test_np = np.array(test)
    x_test = test_np[:, :-1]
    label_test = test_np[:, -1]
    result_list = []

    score = 0
    index = 0
    print('test 개수 :', len(label_test))
    for i in range(len(label_test)):
        # input, output 단어

        # 단어의 index
        input_word_numbers = x_test[i][0]
        output_label = int(label_test[i])

        """""
        w_ins = w_in[input_word_numbers]
        norm_w_ins = (w_ins - np.mean(w_ins, axis=1)) / np.std(w_ins, axis=1)
        sum_w_in = np.sum(norm_w_ins, axis=0)
        """
        sum_w_in = np.sum(w_in[input_word_numbers], axis=0)

        y = softmax(np.dot(w_out.T, sum_w_in))
        result_list.append(np.argmax(y) + 1)
        if np.argmax(y) == output_label:
            score += 1

        index += 1

        if index % 10000 == 0:
            print('testing!!!')

    print("test done!!!")
    print('score :', score / len(label_test))
print("training done!!!")

##########test 결과#################



def save(file_name):
    f = open(file_name, 'w')
    for result in result_list:
        f.write('%s \n' % result)

    f.close()
    print("저장 완료!!!")


save('result.txt')