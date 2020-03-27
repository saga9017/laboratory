
# coding: utf-8

# In[285]:

import os
import re
import time
import random
import numpy as np

# Data Processing and Loading

# Preprocessing
def clean_str(string, TREC=False):
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

# Data Loading
def read_dataset(name):
    # return < [sentence, label]'s list / num_classes >
    # name : Paper's dataset name
    
    ########################################################
    #               < Test type CV >                       #   
    #         return Full_data, num_classes                #
    #                                                      #
    #             < Test type normal >                     # 
    #  return train_data, dev_data, test_data, num_classes #
    ########################################################

    os.chdir(r'C:\Users\조강\Desktop\Yoon Kim CNN\A. Data')
    print('Data : %s' % name)
    if name == 'MR':
        MR = []
        num_classes = 2   # the number of classes
        
        # sentences of positive class
        with open('rt-polaritydata(MR)/rt-polaritydata/rt-polarity.pos.txt','r',encoding='latin-1') as f:
            for row in f.readlines():
                MR.append([clean_str(row),1])
                
        # sentences of negative class
        with open('rt-polaritydata(MR)/rt-polaritydata/rt-polarity.neg.txt','r',encoding='latin-1') as f:
            for row in f.readlines():
                MR.append([clean_str(row),0])
                
        random.shuffle(MR)
        
       
        return MR, num_classes                                  # return Full_data, num_classes
 
    elif name == 'SST-1':
        SST_1_train = []
        SST_1_dev = []
        SST_1_test = []
        num_classes = 5   # the number of classes
        
        # sentences of training data
        with open('SST-1/stsa.fine.train', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_train.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of validation data
        with open('SST-1/stsa.fine.dev', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_dev.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of test data
        with open('SST-1/stsa.fine.test', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_test.append([clean_str_sst(row[1:]),int(row[0])])
                

        return SST_1_train, SST_1_dev, SST_1_test, num_classes  # return train_data, dev_data, test_data, num_classes
    
    elif name == 'SST-2':
        SST_2_train = []
        SST_2_dev = []
        SST_2_test = []
        num_classes = 2   # the number of classes

        # sentences of training data        
        with open('SST-2/stsa.binary.train', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_train.append([clean_str_sst(row[1:]),int(row[0])])

        # sentences of validation data                
        with open('SST-2/stsa.binary.dev', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_dev.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of test data                
        with open('SST-2/stsa.binary.test', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_test.append([clean_str_sst(row[1:]),int(row[0])])
                

        return SST_2_train, SST_2_dev, SST_2_test, num_classes  # return train_data, dev_data, test_data, num_classes
    
    elif name == 'Subj':
        Subj = []
        num_classes = 2   # the number of classes
        
        with open('Subj/subj.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                Subj.append([clean_str(row[1:]),int(row[0])])
        

        return Subj, num_classes                                # return Full_data, num_classes
        
    elif name == 'TREC':
        TREC = []
        TREC_train = []
        TREC_dev = []
        TREC_test = []
        num_classes = 6   # the number of classes

        # sentences of training data   
        with open('TREC/TREC.train.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                TREC.append([clean_str(row[1:], TREC=True),int(row[0])])

        # sentences of test data  
        with open('TREC/TREC.test.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                TREC_test.append([clean_str(row[1:],TREC=True),int(row[0])])
                
        dev_rate = 0.1                # 10% dev
        dev = int(len(TREC)*dev_rate)
        random.shuffle(TREC)
        
        TREC_train = TREC[dev:]         # train data
        TREC_dev = TREC[:dev]           # dev data
        

        return TREC_train, TREC_dev, TREC_test, num_classes    # return train_data, dev_data, test_data, num_classes

    elif name == 'CR':
        CR = []
        num_classes = 2   # the number of classes

        # sentences of Full data 
        with open('CR/custrev.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                CR.append([clean_str(row[1:]),int(row[0])])
                
        print('End reading data %s' % name)
        return CR, num_classes                                  # return Full_data, num_classes
    
    elif name == 'MPQA':
        MPQA = []
        num_classes = 2   # the number of classes
        
        # sentences of Full data 
        with open('MPQA/mpqa.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                MPQA.append([clean_str(row[1:]),int(row[0])])
        

        return MPQA, num_classes                                # return Full_data, num_classes
    
    else:
        print("Error : not available, you have to enter the name exactly") # Error


# In[151]:

class Lang(): # Language set
    def __init__(self,name):
        self.name = name                # Language name
        self.word2index = {'#PAD' : 0}  # Dictionary : Word to index 
        self.index2word = {0 : '#PAD'}  # Dictionary : Index to word
        self.n_words = 1                # Dictionary length
        self.sentences_Full = []        # Language's raw sentences/label (Full set) 
        self.sentences_train = []       # Language's raw sentences/label (training set) 
        self.sentences_dev = []         # Language's raw sentences/label (deviance = validation)
        self.sentences_test = []        # Language's raw sentences/label (test set)

    def AddDictionary(self,sentence):
        for word in sentence.split():
            if not word in self.word2index:
                self.word2index[word]=self.n_words # Add word : index in dictionary
                self.index2word[self.n_words]=word # Add index : word in dictionary
                self.n_words += 1                  # dictionary length

    # Add raw sentences/label (Full set)
    def AddSentence_Full(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_Full.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])

    # Add raw sentences/label (training set)
    def AddSentence_train(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_train.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])
    
    # Add raw sentences/label (deviance set)
    def AddSentence_dev(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_dev.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]]) 
    
    # Add raw sentences/label (test set)
    def AddSentence_test(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_test.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])


# In[283]:

def preprocessing_indexing(name,max_length,CV=True):
    # Test type CV
    if CV==True:
        Data, num_classes = read_dataset(name)      # Full data, the number of classes
        Data_lang = Lang(name)                      # Language set
        for sentence in Data:
            Data_lang.AddDictionary(sentence[0])    # Make dictionary
        Data_lang.AddSentence_Full(Data,max_length) # Full data -> indexing
        
        print(" < return > Sentences/Label by indexing, num_classes, Data's Lang")
        return Data_lang.sentences_Full, num_classes, Data_lang
    
    # Test type not CV
    else:
        Data_train, Data_dev, Data_test, num_classes = read_dataset(name) # training, dev, test, the number of classes
        Data_lang = Lang(name)                                            # Language set
        for Data in [Data_train, Data_dev, Data_test]:
            for sentence in Data:
                Data_lang.AddDictionary(sentence[0])                      # Make dictionary
        Data_lang.AddSentence_train(Data_train,max_length)                # training data -> indexing
        Data_lang.AddSentence_dev(Data_dev,max_length)                    # dev data -> indexing
        Data_lang.AddSentence_test(Data_test,max_length)                  # test data -> indexing
    
        return Data_lang.sentences_train, Data_lang.sentences_dev, Data_lang.sentences_test, num_classes, Data_lang


# In[282]:

def Final_data(name):
    # For entering CNN model inputs setting
    
    max_length = {'MR' : 20*2,
                 'SST-1' : 18*2,
                 'SST-2' : 19*2,
                 'Subj' : 23*2,
                 'TREC' : 10*2,
                 'CR' : 19*2,
                 'MPQA' : 3*2}
    
    if name in ['MR','Subj','CR','MPQA']:
        full, num_classes, lang = preprocessing_indexing(name,max_length[name],CV=True)
        print(" Dataset_size :",len(full),'\n',
             "The number of classes :",num_classes,'\n',
             "Vocabulary size :",len(lang.word2index),'\n')
        return full, num_classes, lang, max_length[name]
    
    else:
        train, dev, test, num_classes, lang = preprocessing_indexing(name,max_length[name],CV=False)
        print(" Dataset_size :",len(train)+len(dev)+len(test),'\n',
             "The number of classes :",num_classes,'\n',
             "Vocabulary size :",len(lang.word2index),'\n')
        return train, dev, test, num_classes, lang, max_length[name]


# In[179]:

## Model


# In[185]:

import torch
import torch.nn as nn
import torch.nn.functional as F

class YoonCNN(nn.Module):
    def __init__(self,**params):
        super(YoonCNN,self).__init__() # initializer
        
        self.Model_type = params['Model_type']                # model type : rand, static, non-static, multichannel
        self.Voca_size = params['Voca_size']                  # vocabulary size
        self.Embedding_size = params['Embedding_size']        # Embedding size
        self.Filter_sizes = params['Filter_sizes']            # sort of filters : [3,4,5]
        self.Num_filters = params['Num_filters']              # the number of filters : 100 
        self.Classes = params['Classes']                      # the number of classes
        self.Length = params['Length']                        # sentence average length times 2
        self.Embedding_weight = params["Embedding_weight"]    # embedding weight init : pretrain or random
  
        
        self.embedding = nn.Embedding(self.Voca_size, self.Embedding_size)                # Embedding
        self.embedding.weight = nn.Parameter(torch.FloatTensor(self.Embedding_weight))    # weight init
        self.embedding.weight.requires_grad=True                                          # the presence or absence
                                                                                            # about Embedding backward

        if self.Model_type == 'static':                     # just update the others parameters(not embedding weight)
            self.embedding.weight.requires_grad=False
            
        if self.Model_type == 'multichannel':               # for concating between two embeddings
            self.mult_embedding = nn.Embedding(self.Voca_size, self.Embedding_size)
            self.mult_embedding.weight = nn.Parameter(torch.FloatTensor(self.Embedding_weight))
            self.mult_embedding.weight.requires_grad=False
                    
        self.conv = nn.ModuleList([nn.Conv2d(1, self.Num_filters, [filter_size, self.Embedding_size],padding=(filter_size-1,0))
                      for filter_size in self.Filter_sizes])                                                   # convolution set
        self.conv[0].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[0], self.Embedding_size)))) # conv1 init
        self.conv[1].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[1], self.Embedding_size)))) # conv2 init
        self.conv[2].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[2], self.Embedding_size)))) # conv3 init
        
        self.fc = nn.Linear(self.Num_filters*len(self.Filter_sizes),self.Classes)                      # fully connected layer
        self.fc.weight = nn.Parameter(torch.FloatTensor(                                               
                np.random.uniform(-0.01,0.01,(self.Classes,self.Num_filters*len(self.Filter_sizes))))) # FC init weight
        self.fc.bias = nn.Parameter(torch.FloatTensor(
                np.zeros(self.Classes)))                                                               # FC init bias(zero)
        
    def forward(self,x):        
        # adding second embedding
        if self.Model_type == 'multichannel':
            Embedding1 = self.embedding(torch.LongTensor(x)).view(-1,1,self.Length,self.Embedding_size)
            Embedding2 = self.mult_embedding(torch.LongTensor(x)).view(-1,1,self.Length,self.Embedding_size)
            Emb = torch.cat((Embedding1,Embedding2),1)
            Emb = Emb.view(-1,1,2*self.Length,self.Embedding_size)
            
        # look up embedding
        else:
            Lookup_Emb = self.embedding(torch.LongTensor(x))
            Emb = torch.unsqueeze(Lookup_Emb,1)

        # CNN
        pooling_output = []
        for conv in self.conv:
            relu = F.relu(conv(Emb))                              # convolution and Relu(activation function)
            relu = torch.squeeze(relu,-1)
            pool = F.max_pool1d(relu,relu.size(2))                # max pooling
            pooling_output.append(pool)
        
        FC_input = torch.cat(pooling_output,2) 
        FC_input = FC_input.view(FC_input.size(0),-1)
        FC_input = F.dropout(FC_input,0.5,training=self.training) # dropout about input
        
        logits = self.fc(FC_input)                                # fully connected layer
        
        probs = F.softmax(logits)                                 # softmax
        classes = torch.max(probs, 1)[1]                          # predict class
        
        return logits, probs, classes


# In[289]:

def train_model(train_data, valid_data=None, test_data=None, CV=True, lr_rate=0.0002, epoches=10):

    if CV == True:                                    # whether to use cross validation or not
        np.random.shuffle(train_data)
        dev_rate = 0.1                                # paper's rate
        valid_rate = int(dev_rate*len(train_data))
        
        CV = 10                                        # 10-Fold
        cv_acc=[]

        for cv in range(CV):
            print(">> CV Step ",cv+1)
            train = train_data[:valid_rate*cv]+train_data[valid_rate*(cv+1):]   # training set
            valid = train_data[valid_rate*cv:valid_rate*(cv+1)]                 # validation set

            model = YoonCNN(**params)                                           # CNN model
            
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
            optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate)      # optimizer :adam


            # training
            for num in range(epoches*int(len(train)/params['Batch_sizes'])+1):
                model.train()                                                   # using dropout
                train_batch = random.sample(train,params['Batch_sizes'])        # mini-batch : 50
                train_input, train_label = zip(*train_batch)

                train_logits, train_probs, train_classes = model(train_input)   # training

                losses = loss_function(train_logits, torch.tensor(train_label)) # calculate loss
                optimizer.zero_grad()                                           # gradient to zero
                losses.backward()                                               # load backward function
                nn.utils.clip_grad_norm_(parameters, 3)                         # clipping threashold : 3
                optimizer.step()                                                # update parameters

                # validation
                if num % int(len(train)/params['Batch_sizes']) == 0:
                    train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                             for i in range(len(train_classes))]), dtype=torch.float)/params['Batch_sizes']


                    model.eval()                                                # not useing dropout
                    valid_input, valid_label = zip(*valid)

                    valid_logits, valid_probs, valid_classes = model(valid_input)
                    valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i] 
                                                 for i in range(len(valid_classes))]),dtype=torch.float)/len(valid)
                    
                    print("Epoch :",int(num/int(len(train)/params['Batch_sizes'])),
                          "-- Loss :",float(losses),
                          " Train_accuracy :",float(train_accuracy),
                          " Valid_accuracy :",float(valid_accuracy))
                
                # k-th cross validation accuracy
                if num == epoches*int(len(train)/params['Batch_sizes']):
                    cv_acc.append(valid_accuracy)
                   
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++","Cross Validation Accuracy :"              ,float(torch.round(torch.sum(torch.tensor(cv_acc))/len(cv_acc)*100000))/100000,'+++++++++++++')
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    else:                                    # whether to use cross validation or not
        model = YoonCNN(**params)            # CNN model

        train = train_data                   # training set
        valid = valid_data                   # validation set
        test = test_data                     # test set
        
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
        optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate)      # optimizer :adam
        
        for num in range(epoches*int(len(train)/params['Batch_sizes'])+1):
            model.train()                                                   # using dropout
            train_batch = random.sample(train,params['Batch_sizes'])        # mini-batch : 50
            train_input, train_label = zip(*train_batch)
            
            train_logits, train_probs, train_classes = model(train_input)   # training
            losses = loss_function(train_logits, torch.tensor(train_label)) # calculate loss
            optimizer.zero_grad()                                           # gradient to zero
            losses.backward()                                               # load backward function
            nn.utils.clip_grad_norm_(parameters, 3)                         # clipping threashold : 3
            optimizer.step()                                                # update parameters
            
            # validation
            if num % int(len(train)/params['Batch_sizes']) == 0:
                train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                         for i in range(len(train_classes))]), dtype=torch.float)/params['Batch_sizes']
                
                model.eval()                                                # not useing dropout
                valid_input, valid_label = zip(*valid)
                
                valid_logits, valid_probs, valid_classes = model(valid_input)
                valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i]
                                                         for i in range(len(valid_classes))]), dtype=torch.float)/len(valid)
                print("Epoch :",int(num/int(len(train)/params['Batch_sizes'])),
                      "-- Loss :",float(losses),
                      " Train_accuracy :",float(train_accuracy),
                      " Valid_accuracy :",float(valid_accuracy))
                
            # test accuracy                
            if num == epoches*int(len(train)/params['Batch_sizes']):
                model.eval()                                                # not useing dropout
                test_input, test_label = zip(*test)
                
                test_logits, test_probs, test_classes = model(test_input)
                test_accuracy = torch.sum(torch.tensor([test_classes[i]==test_label[i] 
                                        for i in range(len(test_classes))]), dtype=torch.float)/len(test)
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("+++++++++++++","Test_accuracy :",float(torch.round(test_accuracy*100000)/100000),"+++++++++++++")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# In[287]:

#import gensim
#model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/조강/Desktop/Yoon Kim CNN/A. Data/GoogleNews-vectors-negative300.bin.gz', binary=True)

def YoonKim_CNN(name,model_type,model_w2v=None,lr_rate=0.0002, epoches=10):

    # Read Data
    if name in ['MR','Subj','CR','MPQA']:
        full, num_classes, lang, length = Final_data(name)
    
    elif name == 'SST-1' or name == 'SST-2' or name == 'TREC':
        train, dev, test, num_classes, lang, length = Final_data(name)
        
    else:
        print("Error : Incorrect Data Name")
    
    # CNN model type setting (embedding weight)
    if model_type == 'rand':
        Embedding_weight=[]
        for word in lang.word2index:
            Embedding_weight.append(np.random.uniform(-0.25,0.25,300).astype("float32"))
    elif model_type == 'static' or model_type == 'non-static' or model_type == 'multichannel':
        Embedding_weight=[]
        for word in lang.word2index:
            if word in model_w2v.wv.vocab:
                Embedding_weight.append(model_w2v.wv[word])
            else:
                Embedding_weight.append(np.random.uniform(-0.25,0.25,300).astype("float32"))
    else:
        print("Error : Incorrect Model Type")
        

    # hyperparamters
    Model_type = model_type
    Voca_size = len(lang.word2index)
    Embedding_size = 300
    Filter_sizes = [3,4,5]
    Num_filters = 100
    Classes = num_classes
    Length = 50
    Embedding_weight = Embedding_weight
    Epoches = 10
    Batch_sizes = 50
    Length = length

    params['Model_type'] = Model_type
    params['Voca_size'] = Voca_size
    params['Embedding_size'] = Embedding_size
    params['Filter_sizes'] = Filter_sizes
    params['Num_filters'] = Num_filters
    params['Classes'] = Classes
    params['Length'] = Length
    params["Embedding_weight"] = Embedding_weight
    params['Batch_sizes'] = Batch_sizes
    params["Length"] = Length
    
    print("Model type :", model_type)
    
    
    # model training / test
    if name in ['MR','Subj','CR','MPQA']:
        train_model(full,CV=True,lr_rate=lr_rate,epoches=epoches)
    else:
        train_model(train,dev,test,CV=False,lr_rate=lr_rate,epoches=epoches)


# In[269]:

import gensim
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('C:/Users/조강/Desktop/Yoon Kim CNN/A. Data/GoogleNews-vectors-negative300.bin.gz', binary=True)


# In[292]:

YoonKim_CNN('SST-1','static',model_w2v)


# In[275]:

###############################################################
################ For using GPU setting cuda ###################
###############################################################
################## torch type.to(device) ######################
###############################################################
from google.colab import drive

drive.mount('/content/drive')

torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:

import os
import re
import time
import random
import numpy as np

# Data Processing and Loading

# Preprocessing
def clean_str(string, TREC=False):
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
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

# Data Loading
def read_dataset(name):
    # return < [sentence, label]'s list / num_classes >
    # name : Paper's dataset name
    
    ########################################################
    #               < Test type CV >                       #   
    #         return Full_data, num_classes                #
    #                                                      #
    #             < Test type normal >                     # 
    #  return train_data, dev_data, test_data, num_classes #
    ########################################################


    print('Data : %s' % name)
    if name == 'MR':
        MR = []
        num_classes = 2   # the number of classes
        
        # sentences of positive class
        with open('/content/drive/My Drive/rt-polarity.pos.txt','r',encoding='latin-1') as f:
            for row in f.readlines():
                MR.append([clean_str(row),1])
                
        # sentences of negative class
        with open('/content/drive/My Drive/rt-polarity.neg.txt','r',encoding='latin-1') as f:
            for row in f.readlines():
                MR.append([clean_str(row),0])
                
        random.shuffle(MR)
        
       
        return MR, num_classes                                  # return Full_data, num_classes
 
    elif name == 'SST-1':
        SST_1_train = []
        SST_1_dev = []
        SST_1_test = []
        num_classes = 5   # the number of classes
        
        # sentences of training data
        with open('/content/drive/My Drive/stsa.finetrain.txt', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_train.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of validation data
        with open('/content/drive/My Drive/stsa.finedev.txt', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_dev.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of test data
        with open('/content/drive/My Drive/stsa.finetest.txt', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_1_test.append([clean_str_sst(row[1:]),int(row[0])])
                

        return SST_1_train, SST_1_dev, SST_1_test, num_classes  # return train_data, dev_data, test_data, num_classes
    
    elif name == 'SST-2':
        SST_2_train = []
        SST_2_dev = []
        SST_2_test = []
        num_classes = 2   # the number of classes

        # sentences of training data        
        with open('/content/drive/My Drive/stsa.binary.train', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_train.append([clean_str_sst(row[1:]),int(row[0])])

        # sentences of validation data                
        with open('/content/drive/My Drive/stsa.binary.dev', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_dev.append([clean_str_sst(row[1:]),int(row[0])])
                
        # sentences of test data                
        with open('/content/drive/My Drive/stsa.binary.test', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                SST_2_test.append([clean_str_sst(row[1:]),int(row[0])])
                

        return SST_2_train, SST_2_dev, SST_2_test, num_classes  # return train_data, dev_data, test_data, num_classes
    
    elif name == 'Subj':
        Subj = []
        num_classes = 2   # the number of classes
        
        with open('/content/drive/My Drive/subj.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                Subj.append([clean_str(row[1:]),int(row[0])])
        

        return Subj, num_classes                                # return Full_data, num_classes
        
    elif name == 'TREC':
        TREC = []
        TREC_train = []
        TREC_dev = []
        TREC_test = []
        num_classes = 6   # the number of classes

        # sentences of training data   
        with open('/content/drive/My Drive/TREC.train.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                TREC.append([clean_str(row[1:], TREC=True),int(row[0])])

        # sentences of test data  
        with open('/content/drive/My Drive/TREC.test.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                TREC_test.append([clean_str(row[1:],TREC=True),int(row[0])])
                
        dev_rate = 0.1                # 10% dev
        dev = int(len(TREC)*dev_rate)
        random.shuffle(TREC)
        
        TREC_train = TREC[dev:]         # train data
        TREC_dev = TREC[:dev]           # dev data
        

        return TREC_train, TREC_dev, TREC_test, num_classes    # return train_data, dev_data, test_data, num_classes

    elif name == 'CR':
        CR = []
        num_classes = 2   # the number of classes

        # sentences of Full data 
        with open('/content/drive/My Drive/custrev.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                CR.append([clean_str(row[1:]),int(row[0])])
                

        return CR, num_classes                                  # return Full_data, num_classes
    
    elif name == 'MPQA':
        MPQA = []
        num_classes = 2   # the number of classes
        
        # sentences of Full data 
        with open('/content/drive/My Drive/mpqa.all', 'r', encoding='latin-1') as f:
            for row in f.readlines():
                MPQA.append([clean_str(row[1:]),int(row[0])])
        

        return MPQA, num_classes                                # return Full_data, num_classes
    
    else:
        print("Error : not available, you have to enter the name exactly") # Error


# In[ ]:

class Lang(): # Language set
    def __init__(self,name):
        self.name = name                # Language name
        self.word2index = {'#PAD' : 0}  # Dictionary : Word to index 
        self.index2word = {0 : '#PAD'}  # Dictionary : Index to word
        self.n_words = 1                # Dictionary length
        self.sentences_Full = []        # Language's raw sentences/label (Full set) 
        self.sentences_train = []       # Language's raw sentences/label (training set) 
        self.sentences_dev = []         # Language's raw sentences/label (deviance = validation)
        self.sentences_test = []        # Language's raw sentences/label (test set)

    def AddDictionary(self,sentence):
        for word in sentence.split():
            if not word in self.word2index:
                self.word2index[word]=self.n_words # Add word : index in dictionary
                self.index2word[self.n_words]=word # Add index : word in dictionary
                self.n_words += 1                  # dictionary length

    # Add raw sentences/label (Full set)
    def AddSentence_Full(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_Full.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])

    # Add raw sentences/label (training set)
    def AddSentence_train(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_train.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])
    
    # Add raw sentences/label (deviance set)
    def AddSentence_dev(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_dev.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]]) 
    
    # Add raw sentences/label (test set)
    def AddSentence_test(self,data,max_length):
        for row in data:
            edit=[]
            for word in row[0].split():
                edit.append(self.word2index[word])
            self.sentences_test.append([edit[:max_length]+[0]*(max_length-len(edit)),row[1]])


# In[ ]:

def preprocessing_indexing(name,max_length,CV=True):
    # Test type CV
    if CV==True:
        Data, num_classes = read_dataset(name)      # Full data, the number of classes
        Data_lang = Lang(name)                      # Language set
        for sentence in Data:
            Data_lang.AddDictionary(sentence[0])    # Make dictionary
        Data_lang.AddSentence_Full(Data,max_length) # Full data -> indexing
        
        print(" < return > Sentences/Label by indexing, num_classes, Data's Lang")
        return Data_lang.sentences_Full, num_classes, Data_lang
    
    # Test type not CV
    else:
        Data_train, Data_dev, Data_test, num_classes = read_dataset(name) # training, dev, test, the number of classes
        Data_lang = Lang(name)                                            # Language set
        for Data in [Data_train, Data_dev, Data_test]:
            for sentence in Data:
                Data_lang.AddDictionary(sentence[0])                      # Make dictionary
        Data_lang.AddSentence_train(Data_train,max_length)                # training data -> indexing
        Data_lang.AddSentence_dev(Data_dev,max_length)                    # dev data -> indexing
        Data_lang.AddSentence_test(Data_test,max_length)                  # test data -> indexing
    
        return Data_lang.sentences_train, Data_lang.sentences_dev, Data_lang.sentences_test, num_classes, Data_lang


# In[ ]:

def Final_data(name):
    # For entering CNN model inputs setting
    
    max_length = {'MR' : 20*2,
                 'SST-1' : 18*2,
                 'SST-2' : 19*2,
                 'Subj' : 23*2,
                 'TREC' : 10*2,
                 'CR' : 19*2,
                 'MPQA' : 3*2}
    
    if name in ['MR','Subj','CR','MPQA']:
        full, num_classes, lang = preprocessing_indexing(name,max_length[name],CV=True)
        print(" Dataset_size :",len(full),'\n',
             "The number of classes :",num_classes,'\n',
             "Vocabulary size :",len(lang.word2index),'\n')
        return full, num_classes, lang, max_length[name]
    
    else:
        train, dev, test, num_classes, lang = preprocessing_indexing(name,max_length[name],CV=False)
        print(" Dataset_size :",len(train)+len(dev)+len(test),'\n',
             "The number of classes :",num_classes,'\n',
             "Vocabulary size :",len(lang.word2index),'\n')
        return train, dev, test, num_classes, lang, max_length[name]


# In[ ]:

import torch
import torch.nn as nn
import torch.nn.functional as F

class YoonCNN(nn.Module):
    def __init__(self,**params):
        super(YoonCNN,self).__init__() # initializer
        
        self.Model_type = params['Model_type']                # model type : rand, static, non-static, multichannel
        self.Voca_size = params['Voca_size']                  # vocabulary size
        self.Embedding_size = params['Embedding_size']        # Embedding size
        self.Filter_sizes = params['Filter_sizes']            # sort of filters : [3,4,5]
        self.Num_filters = params['Num_filters']              # the number of filters : 100 
        self.Classes = params['Classes']                      # the number of classes
        self.Length = params['Length']                        # sentence average length times 2
        self.Embedding_weight = params["Embedding_weight"]    # embedding weight init : pretrain or random
        
        
        self.embedding = nn.Embedding(self.Voca_size, self.Embedding_size)                # Embedding
        self.embedding.weight = nn.Parameter(torch.FloatTensor(self.Embedding_weight))    # weight init
        self.embedding.weight.requires_grad=True                                          # the presence or absence
                                                                                            # about Embedding backward

        if self.Model_type == 'static':                     # just update the others parameters(not embedding weight)
            self.embedding.weight.requires_grad=False
            
        if self.Model_type == 'multichannel':               # for concating between two embeddings
            self.mult_embedding = nn.Embedding(self.Voca_size, self.Embedding_size)
            self.mult_embedding.weight = nn.Parameter(torch.FloatTensor(self.Embedding_weight))
            self.mult_embedding.weight.requires_grad=False
                    
        self.conv = nn.ModuleList([nn.Conv2d(1, self.Num_filters, [filter_size, self.Embedding_size],padding=(filter_size-1,0))
                      for filter_size in self.Filter_sizes])                                                   # convolution set
        self.conv[0].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[0], self.Embedding_size)))) # conv1 init
        self.conv[1].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[1], self.Embedding_size)))) # conv2 init
        self.conv[2].weight = nn.Parameter(torch.FloatTensor(
                np.random.uniform(-0.01,0.01,(self.Num_filters,1,self.Filter_sizes[2], self.Embedding_size)))) # conv3 init
        
        self.fc = nn.Linear(self.Num_filters*len(self.Filter_sizes),self.Classes)                      # fully connected layer
        self.fc.weight = nn.Parameter(torch.FloatTensor(                                               
                np.random.uniform(-0.01,0.01,(self.Classes,self.Num_filters*len(self.Filter_sizes))))) # FC init weight
        self.fc.bias = nn.Parameter(torch.FloatTensor(
                np.zeros(self.Classes)))                                                               # FC init bias(zero)
        
    def forward(self,x):        
        # adding second embedding
        if self.Model_type == 'multichannel':
            Embedding1 = self.embedding(torch.LongTensor(x).to(device)).view(-1,1,self.Length,self.Embedding_size)
            Embedding2 = self.mult_embedding(torch.LongTensor(x).to(device)).view(-1,1,self.Length,self.Embedding_size)
            Emb = torch.cat((Embedding1,Embedding2),1)
            Emb = Emb.view(-1,1,2*self.Length,self.Embedding_size)
            
        # look up embedding
        else:
            Lookup_Emb = self.embedding(torch.LongTensor(x).to(device))
            Emb = torch.unsqueeze(Lookup_Emb,1)

                          
        pooling_output = []
        for conv in self.conv:
            relu = F.relu(conv(Emb))                              # convolution and Relu(activation function)
            relu = torch.squeeze(relu,-1)
            pool = F.max_pool1d(relu,relu.size(2))                # max pooling
            pooling_output.append(pool)
        
        FC_input = torch.cat(pooling_output,2)
        FC_input = FC_input.view(FC_input.size(0),-1)
        FC_input = F.dropout(FC_input,0.5,training=self.training) # dropout about input
        
        logits = self.fc(FC_input)                                # fully connected layer
        
        probs = F.softmax(logits)                                 # softmax
        classes = torch.max(probs, 1)[1]                          # predict class
        
        return logits, probs, classes


# In[ ]:

def train_model(train_data, valid_data=None, test_data=None, CV=True, lr_rate=0.0002, epoches=10):
    print("Modeling Start.....")
    if CV == True:                                    # whether to use cross validation or not
        np.random.shuffle(train_data)
        dev_rate = 0.1                                # paper's rate
        valid_rate = int(dev_rate*len(train_data))
        
        CV = 10                                        # 10-Fold
        cv_acc=[]

        for cv in range(CV): 
            print(">> CV Step ",cv+1)
            train = train_data[:valid_rate*cv]+train_data[valid_rate*(cv+1):]   # training set
            valid = train_data[valid_rate*cv:valid_rate*(cv+1)]                 # validation set

            model = YoonCNN(**params).to(device)                                # CNN model (to device)
            
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
            optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate)      # optimizer :adam


            for num in range(epoches*int(len(train)/params['Batch_sizes'])+1):
                model.train()                                                   # using dropout
                train_batch = random.sample(train,params['Batch_sizes'])        # mini-batch : 50
                train_input, train_label = zip(*train_batch)

                train_logits, train_probs, train_classes = model(train_input)   # training

                losses = loss_function(train_logits, torch.tensor(train_label).to(device)) # calculate loss (to device)
                optimizer.zero_grad()                                                      # gradient to zero
                losses.backward()                                                          # load backward function
                nn.utils.clip_grad_norm_(parameters, 3)                                    # clipping threashold : 3
                optimizer.step()                                                           # update parameters

                # validation
                if num % int(len(train)/params['Batch_sizes']) == 0:
                    train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                             for i in range(len(train_classes))]).to(device), dtype=torch.float)/params['Batch_sizes']


                    model.eval()                                                 # not useing dropout
                    valid_input, valid_label = zip(*valid)

                    valid_logits, valid_probs, valid_classes = model(valid_input)
                    valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i] 
                                                 for i in range(len(valid_classes))]).to(device),dtype=torch.float)/len(valid)
                    print("Epoch :",int(num/int(len(train)/params['Batch_sizes'])),
                          "-- Loss :",float(losses),
                          " Train_accuracy :",float(train_accuracy),
                          " Valid_accuracy :",float(valid_accuracy))
                
                # k-th cross validation accuracy
                if num == epoches*int(len(train)/params['Batch_sizes']):
                    cv_acc.append(valid_accuracy)
                   
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++","Cross Validation Accuracy :"              ,float(torch.round(torch.sum(torch.tensor(cv_acc).to(device))/len(cv_acc)*100000))/100000,'+++++++++++++')
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    else:                                               # whether to use cross validation or not
        model = YoonCNN(**params).to(device)            # CNN model (to device)

        train = train_data                              # training set
        valid = valid_data                              # validation set
        test = test_data                                # test set
        
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        loss_function = nn.CrossEntropyLoss()                               # loss function : Cross Entropy
        optimizer = torch.optim.Adam(model.parameters(), lr = lr_rate)      # optimizer :adam
        
        for num in range(epoches*int(len(train)/params['Batch_sizes'])+1):
            model.train()                                                   # using dropout
            train_batch = random.sample(train,params['Batch_sizes'])        # mini-batch : 50
            train_input, train_label = zip(*train_batch)
            
            train_logits, train_probs, train_classes = model(train_input)              # training
            losses = loss_function(train_logits, torch.tensor(train_label).to(device)) # calculate loss (to device)
            optimizer.zero_grad()                                                      # gradient to zero
            losses.backward()                                                          # load backward function
            nn.utils.clip_grad_norm_(parameters, 3)                                    # clipping threashold : 3
            optimizer.step()                                                           # update parameters
            
            if num % int(len(train)/params['Batch_sizes']) == 0:
                train_accuracy = torch.sum(torch.tensor([train_classes[i]==train_label[i] 
                                         for i in range(len(train_classes))]).to(device), dtype=torch.float)/params['Batch_sizes']
                
                model.eval()                                                            # not useing dropout
                valid_input, valid_label = zip(*valid)
                
                valid_logits, valid_probs, valid_classes = model(valid_input)
                valid_accuracy = torch.sum(torch.tensor([valid_classes[i]==valid_label[i]
                                                         for i in range(len(valid_classes))]).to(device), dtype=torch.float)/len(valid)
                print("Epoch :",int(num/int(len(train)/params['Batch_sizes'])),
                      "-- Loss :",float(losses),
                      " Train_accuracy :",float(train_accuracy),
                      " Valid_accuracy :",float(valid_accuracy))
                
                
            if num == epoches*int(len(train)/params['Batch_sizes']):
                model.eval()                                                            # not useing dropout
                test_input, test_label = zip(*test)
                
                test_logits, test_probs, test_classes = model(test_input)
                test_accuracy = torch.sum(torch.tensor([test_classes[i]==test_label[i] 
                                        for i in range(len(test_classes))]).to(device), dtype=torch.float)/len(test)
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                print("+++++++++++++","Test_accuracy :",float(torch.round(test_accuracy*100000)/100000),"+++++++++++++")
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# In[ ]:

import gensim
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin.gz', binary=True)

def YoonKim_CNN(name,model_type,model_w2v=None,lr_rate=0.0002, epoches=10):

    # Read data
    if name in ['MR','Subj','CR','MPQA']:
        full, num_classes, lang, length = Final_data(name)
    
    elif name == 'SST-1' or name == 'SST-2' or name == 'TREC':
        train, dev, test, num_classes, lang, length = Final_data(name)
        
    else:
        print("Error : Incorrect Data Name")
    
    # CNN model type setting (embedding weight)
    if model_type == 'rand':
        Embedding_weight=[]
        for word in lang.word2index:
            Embedding_weight.append(np.random.uniform(-0.25,0.25,300).astype("float32"))
    elif model_type == 'static' or model_type == 'non-static' or model_type == 'multichannel':
        Embedding_weight=[]
        for word in lang.word2index:
            if word in model_w2v.wv.vocab:
                Embedding_weight.append(model_w2v.wv[word])
            else:
                Embedding_weight.append(np.random.uniform(-0.25,0.25,300).astype("float32"))
    else:
        print("Error : Incorrect Model Type")
        

    # hyperparameters
    Model_type = model_type
    Voca_size = len(lang.word2index)
    Embedding_size = 300
    Filter_sizes = [3,4,5]
    Num_filters = 100
    Classes = num_classes
    Length = 50
    Embedding_weight = Embedding_weight
    Epoches = 10
    Batch_sizes = 50
    Length = length

    params['Model_type'] = Model_type
    params['Voca_size'] = Voca_size
    params['Embedding_size'] = Embedding_size
    params['Filter_sizes'] = Filter_sizes
    params['Num_filters'] = Num_filters
    params['Classes'] = Classes
    params['Length'] = Length
    params["Embedding_weight"] = Embedding_weight
    params['Batch_sizes'] = Batch_sizes
    params["Length"] = Length
    
    print("Model type :", model_type)
    
    # model training / test
    if name in ['MR','Subj','CR','MPQA']:
        train_model(full,CV=True,lr_rate=lr_rate,epoches=epoches)
    else:
        train_model(train,dev,test,CV=False,lr_rate=lr_rate,epoches=epoches)


# In[ ]:

YoonKim_CNN('SST-2','multichannel',model_w2v)

