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

pretrained_weights = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
# Models can return full list of hidden-states & attentions weights at each layer
bert=BertModel.from_pretrained(pretrained_weights,
                                    output_hidden_states=True,
                                    output_attentions=True)


ITERATION=200
MAX_LENGTH=150

mask={}
for i in range(MAX_LENGTH+1):
    if torch.cuda.is_available()==True:
        mask[i + 1] = torch.triu(torch.ones((i + 1, i + 1)), diagonal=1).bool()
    else:
        mask[i+1] = torch.triu(torch.ones((i+1, i+1)), diagonal=1).byte()

def Positional_Encoding(MAX_LENGTH, hidden_dim):  # x : (batch_size, input_len, hidden_dim) or (batch_size, output_len, hidden_dim)
    table = torch.zeros((MAX_LENGTH, hidden_dim))
    a, b = table.shape
    for pos in range(a):
        for i in range(b):
            if i % 2 == 0:
                table[pos][i] = math.sin(pos / math.pow(10000, i / b))
            else:
                table[pos][i] = math.cos(pos / math.pow(10000, (i - 1) / b))
    return Parameter(table, requires_grad=False)


#################################################### Module ############################################################

class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()
        self.gamma=Parameter(torch.tensor(1.0))
        self.beta = Parameter(torch.tensor(0.0))
    def forward(self, X):
        mu = torch.mean(X, dim=2)
        var = torch.var(X, dim=2)
        X_norm = torch.div(X - mu.view(X.shape[0], X.shape[1], 1), torch.sqrt(var.view(X.shape[0], X.shape[1], 1) + 1e-8))
        out = self.gamma * X_norm + self.beta
        return out

class Multi_head_attention(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.layerNorm_add_Norm = Norm()

        self.w_qs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_ks = nn.Linear(hidden_dim, hidden_dim_)
        self.w_vs = nn.Linear(hidden_dim, hidden_dim_)
        self.w_os = nn.Linear(hidden_dim_, hidden_dim)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (hidden_dim)))
        nn.init.xavier_normal_(self.w_os.weight)

    def forward(self, en, de, mask, dropout=False):  # x : (input_len, hidden_dim)
        d_k = de.shape[-1]
        len_d0, len_d1=de.shape[0], de.shape[1]
        len_e0, len_e1=en.shape[0], en.shape[1]


        q = self.w_qs(de).view(len_d0, len_d1, -1, 8).permute(3, 0, 1, 2)
        k = self.w_ks(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 2, 1)
        v = self.w_vs(en).view(len_e0, len_e1, -1, 8).permute(3, 0, 1, 2)

        e = torch.matmul(q, k) / math.sqrt(d_k)
        masked_e = e.masked_fill(mask, -1e10)
        alpha = self.softmax(masked_e)  # (output_len, input_len)
        if dropout==True:
            alpha = self.dropout(alpha)
        head3 = torch.matmul(alpha, v)

        a = torch.cat((head3[0], head3[1], head3[2], head3[3], head3[4], head3[5], head3[6], head3[7]), 2)

        result = self.w_os(a)
        result=self.layerNorm_add_Norm(result+de)
        return result  # (output_len, hidden)

class FFN(nn.Module):  # feed forward network   x : (batch_size, input_len, hidden)
    def __init__(self, hidden_dim=300, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm_add_Norm = Norm()

        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4*hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


    def forward(self, x, dropout=False):
        output = self.fc1(x) # (batch_size, input_len, 4*hidden)
        if dropout==True:
            output = self.dropout(F.relu(output))  # (batch_size, input_len, 4*hidden)
        else:
            output = F.relu(output)
        output = self.fc2(output)  # (batch_size, input_len, hidden
        output=self.layerNorm_add_Norm(output+x)
        return output

##################################################### Sub layer ########################################################

class Decoder_layer(nn.Module):
    def __init__(self, hidden_dim=300, hidden_dim_=512, dropout=0.1):  # default=512
        # Assign instance variables
        super().__init__()

        self.masked_multi_head_self_attention = Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.multi_head_attention=Multi_head_attention(hidden_dim, hidden_dim_, dropout)
        self.ffn=FFN(hidden_dim, dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)


    def forward(self, y, en, mask1, mask2, non_pad_mask_y=1, dropout=False):
        if dropout==True:
            output = self.dropout_1(self.masked_multi_head_self_attention(y, y, mask1, dropout=True))
            output = output*non_pad_mask_y
            output = self.dropout_2(self.multi_head_attention(en, output, mask2, dropout=True))
            output = output*non_pad_mask_y
            output = self.dropout_3(self.ffn(output, dropout=True))
            output = output*non_pad_mask_y
        else:
            output = self.masked_multi_head_self_attention(y, y, mask1)
            output = output * non_pad_mask_y
            output = self.multi_head_attention(en, output, mask2)
            output = output * non_pad_mask_y
            output = self.ffn(output)
            output = output * non_pad_mask_y
        return output


class Decoder_for_generation(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Decoder_for_generation, self).__init__()
        self.hidden=768
        self.embedding = copy.deepcopy(bert).embeddings
        ########### Option : bert embedding Freeze####################
        self.embedding.require_grad=False
        self.transformer_decoders = nn.ModuleList(
            [Decoder_layer(hidden_dim=768, hidden_dim_=768, dropout=0.1) for _ in range(6)])
        self.softmax=nn.Softmax(-1)
        self.V_d = nn.Linear(768, 10)    # orinin= tokenizer.vocab_size
        nn.init.xavier_normal_(self.V_d.weight)

    def output_embedding(self, y):  # x: (batch_size, output_len, )
        mask = (y == 0)
        return self.embedding(y), mask  # (output_len, hidden_dim)

    def forward(self, last_encoder_output, mask_d, y, dropout):
        y1, mask_dd = self.output_embedding(y)  # (output_len,hidden)
        ################### MASKs ########################################################
        non_pad_mask_y = (y != 0).unsqueeze(-1).repeat(1, 1, self.hidden).float()
        mask_dd = mask_dd.unsqueeze(1).repeat(1, y.shape[1], 1) | mask[len(y[0])].to(device)
        mask_d = mask_d.unsqueeze(1).repeat(1, y.shape[1], 1)
        ##################################################################################
        output = y1.masked_fill(non_pad_mask_y == 0, 0)
        ########################################################
        residual=0
        for index, decoder_layer in enumerate(self.transformer_decoders):
            if index%2==0:
                output=decoder_layer(output+residual, last_encoder_output, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
                residual=output
            else:
                output = decoder_layer(output, last_encoder_output, mask_dd, mask_d, non_pad_mask_y, dropout=dropout)
        #######################################################
        output = self.softmax(self.V_d(output))
        return output


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
class Bert_for_generation(nn.Module):
    def __init__(self, label_smoothing=0.1):
        # Assign instance variables
        super(Bert_for_generation, self).__init__()
        self.hidden=768
        self.encoder=Encoder_for_generation()
        ########### bert model Freeze ####################
        self.encoder.require_grad=False
        self.decoder=Decoder_for_generation()

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
model=Bert_for_generation().to(device)
last_loss = model.train_with_batch(batch)

predicts=[]
for i in range(len(batch)):
    predicts.append(model.predict(batch[i][0]))

    if i%10==0:
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time, ' ', 100*i/len(batch), '%   저장!!!')

print(predicts)