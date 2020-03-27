import torch
import torch.nn as nn
import numpy as np
import copy
import math
import torch.nn.functional  as F
from transformers import *
from datetime import datetime
from torch.nn.parameter import Parameter


print('gpu :', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pretrained_weights = 'bert-base-multilingual-uncased'
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# Models can return full list of hidden-states & attentions weights at each layer
bert=BertModel.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)


ITERATION=200
MAX_LENGTH=150


class Encoder_for_generation(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Encoder_for_generation, self).__init__()
        self.bert=copy.deepcopy(bert)


    def forward(self, x):
        all_hidden_states, _ = self.bert(x)[-2:]
        last_encoder_output = all_hidden_states[-1][:, 1:]
        mask_d=(x==0)[:,1:]
        return last_encoder_output, mask_d

########################################## My Model #######################################################
class Bert_for_classification(nn.Module):
    def __init__(self, label_smoothing=0.1):
        # Assign instance variables
        super(Bert_for_classification, self).__init__()
        self.hidden=768
        self.encoder=Encoder_for_generation()
        ########### bert model Freeze ####################
        self.encoder.require_grad=False

        self.target_prob = (1 - label_smoothing) + label_smoothing / tokenizer.vocab_size
        self.nontarget_prob =label_smoothing / tokenizer.vocab_size


        self.Loss=0

    def forward(self, x, y):
        last_encoder_output, mask_d = self.encoder(x)
        output = self.decoder(last_encoder_output, mask_d, y, dropout=True)
        return output

    def bptt(self, x, y, y_):  # (batch_size, out_len)
        y9 = self.forward(x, y)
        a,b=y_.nonzero().t()[0], y_.nonzero().t()[1]
        z=y9[a,b]
        pos=torch.log(z.gather(1, y_[a,b].unsqueeze(-1))).squeeze()
        neg=torch.sum(torch.log(z), dim=1)-pos
        loss = -self.target_prob * pos - self.nontarget_prob * neg
        loss=torch.mean(loss)
        return loss

    def predict(self, x,):
        x=torch.tensor(x).to(device)
        self.encoder.eval()
        ###############################################################################################################
        last_encoder_output, mask_d=self.encoder(x)
        ###############################################################################################################
        output = torch.tensor([[1]] * x.shape[0]).to(device)
        step = 0
        while step < MAX_LENGTH+1:
            y=self.decoder(last_encoder_output, mask_d, output, dropout=False)
            output = torch.cat((output, torch.argmax(y[:,-1], dim=1).unsqueeze(-1)), 1)
            step += 1
        return output

    def numpy_sdg_step(self, batch, optimizer):
        # Calculate the gradients

        optimizer.zero_grad()
        loss = self.bptt(torch.tensor(batch[0]).to(device), batch[1], torch.tensor(batch[2]).to(device))
        loss.backward()
        optimizer.step()
        """""
        optimizer.param_groups[0]['lr'] = math.pow(self.hidden, -0.5) \
                                          * min(math.pow(num_examples_seen, -0.5),
                                                num_examples_seen * math.pow(10, -1.5))  # warm up step : default 4000
        """""

        return loss

    def train_with_batch(self, batch):
        # We keep track of the losses so we can plot them later
        global num_examples_seen
        num_examples_seen = 1
        Loss_len = 0
        last_loss=0
        nepoch=int(ITERATION/len(batch))
        print('epoch :', nepoch)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=5e-5)

        for epoch in range(nepoch):
            # For each training example...
            for i in range(len(batch)):
                # One SGD step
                self.Loss += self.numpy_sdg_step(batch[i], optimizer).item()

                Loss_len += 1
                if num_examples_seen == len(batch)*nepoch:
                    last_loss= self.Loss / Loss_len
                else:
                    if int(len(batch) * nepoch/100)==0:
                        time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                        print('%   완료!!!', end='')
                        print('   loss :', self.Loss / Loss_len)
                        self.Loss = 0
                        Loss_len = 0
                    else:
                        if num_examples_seen %  int(len(batch) * nepoch/100)== 0:
                            time_ = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            print(time_, ' ', int(100 * num_examples_seen / (len(batch) * nepoch)), end='')
                            print('%   완료!!!', end='')
                            print('   loss :', self.Loss / Loss_len)
                            self.Loss = 0
                            Loss_len = 0
                num_examples_seen += 1
        return last_loss

X=torch.tensor([[tokenizer.cls_token_id]+tokenizer.encode("나는 한국인 이다."), [tokenizer.cls_token_id]+tokenizer.encode("나는 일본인 이다.")])
Y=torch.tensor([[1,5,6,7,0,0,0,0,0,0], [1,8,9,0,0,0,0,0,0,0]]).to(device)
Y_=torch.tensor([[5,6,7,2,0,0,0,0,0,0], [8,9,2,0,0,0,0,0,0,0]]).to(device)

batch=[(X,Y,Y_)]
model=Bert_for_classification().to(device)
last_loss = model.train_with_batch(batch)

predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0]))

    if i%10==0:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, ' ', 100*i/len(batch), '%   저장!!!')

print(predicts)