"""""""""""""""""""""""""""""""""""""""""
1. torch library를 설치하셔야합니다.(google 검색 참고)
2. data를 excel 파일형식으로 저장하셔야 합니다.
3. data를 섞어 80%는 project2_train.xlsx로 저장하고, 10%는 project2_dev.xlsx, 나머지 10%는 project2_test.xlsx로 저장합니다. (현재는 train, dev, test가 모두 같은 파일입니다.)
4. 이 data파일은 python 실행파일 안에 같이 있어야 합니다.
5. generate_batch 함수를 data와 feature에 맞게 수정하셔야 합니다.(사용하시고 싶은 feature들을 추가하셔도 됩니다.)
6. epoch, optimizer의 lr이 각각 10, 0.01으로 고정되어 있는데, 학습정도를 보고 줄이거나 늘리셔도 됩니다.
7. 실행시키고 나면 'saved_model_epoch=?' 파일이 여러개 만들어 지는데 우선 무시하셔도 됩니다.(후에 model을 다시 불러올 때 사용합니다.)
첫번째 방법은 feature를 사람이 직접 구간으로 나누는 방법입니다. 예를 들어, 예수금이 <1억 이면 1, 아니면 2 이런 식으로 구간을 나눕니다.
"""""""""""""""""""""""""""""""""""""""""
from openpyxl import load_workbook
import torch
import torch.nn.functional  as F
import torch.nn as nn


def generate_batch(file_name):
    load_wb = load_workbook(file_name, data_only=True)
    load_ws = load_wb['Sheet1']

    used_index=[3, 7, 8, 9, 10 ,11, 12, 13, 14]

    #data 전처리
    X=[]
    y=[]
    for index2, row in enumerate(load_ws.rows):
        if index2==0:
            continue
        temp=[]
        for index, data in enumerate(row):
            if index in used_index:
                if index==3:
                    if data.value=='Y':
                        temp.append(1)
                    else:
                        temp.append(0)
                elif index==7:
                    temp.append(int(data.value/10))
                elif index==8:
                    if data.value=='남':
                        temp.append(1)
                    else:
                        temp.append(0)
                elif index==14:
                    if data.value=='R':
                        temp.append(2)
                    elif data.value=='A':
                        temp.append(1)
                    else:
                        temp.append(0)
                elif index==10:
                    if data.value<20000200:
                        temp.append(1)
                    else:
                        temp.append(0)
                elif index==11:
                    if data.value=='-':
                        temp.append(0)
                    else:
                        if data.value<5000000:
                            temp.append(2)
                        else:
                            temp.append(1)
                elif index==12:
                    if data.value == '-':
                        temp.append(0)
                    else:
                        if data.value < 100000000:
                            temp.append(2)
                        else:
                            temp.append(1)
                elif index==13:
                    if data.value=='-':
                        temp.append(0)
                    else:
                        if data.value<200000000:
                            temp.append(2)
                        else:
                            temp.append(1)
                else:
                    if data.value=='-':
                        temp.append(0)
                    else:
                        temp.append(data.value)
        X.append(temp[:-1])
        y.append(temp[-1])
    return X, y



class classification(nn.Module):
    def __init__(self):  # default=512
        # Assign instance variables
        super().__init__()
        self.embeddings=nn.ModuleList([nn.Embedding(2, 128), nn.Embedding(9, 128), nn.Embedding(2, 128), nn.Embedding(21421, 128),
                                       nn.Embedding(2, 128), nn.Embedding(3, 128), nn.Embedding(3, 128), nn.Embedding(3, 128)])
        for i in self.embeddings:
            i.weight.data.uniform_(-0.1, 0.1)

        self.conv1 = nn.Conv2d(1, 100, (3, 128))
        self.conv2 = nn.Conv2d(1, 100, (4, 128))
        self.conv3 = nn.Conv2d(1, 100, (5, 128))
        ############### weight initialization ##################
        self.conv1.weight.data.uniform_(-0.01, 0.01)
        self.conv2.weight.data.uniform_(-0.01, 0.01)
        self.conv3.weight.data.uniform_(-0.01, 0.01)

        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.1)
        self.dropout_3 = nn.Dropout(0.1)

        self.linear = nn.Linear(300, 3)
        nn.init.xavier_normal_(self.linear.weight)


        self.softmax=nn.Softmax(-1)

    def forward(self, x, dropout):
        stack=[]
        for a,b  in zip(x, self.embeddings):
            stack.append(b(a).unsqueeze(0))
        result=torch.cat(stack, dim=0).view(1,1, -1 ,128)
        if dropout==True:
            result1=self.dropout_1(F.max_pool1d(F.relu(self.conv1(result).squeeze(-1)), 6).squeeze())
            result2=self.dropout_2(F.max_pool1d(F.relu(self.conv2(result).squeeze(-1)), 5).squeeze())
            result3=self.dropout_3(F.max_pool1d(F.relu(self.conv3(result).squeeze(-1)), 4).squeeze())
        else:
            result1 = F.max_pool1d(F.relu(self.conv1(result).squeeze(-1)), 6).squeeze()
            result2 = F.max_pool1d(F.relu(self.conv2(result).squeeze(-1)), 5).squeeze()
            result3 = F.max_pool1d(F.relu(self.conv3(result).squeeze(-1)), 4).squeeze()

        result=torch.cat([result1, result2, result3], 0)

        result=self.softmax(self.linear(result))

        return result

    def predict(self, x):
        output=self.forward(x, False)
        output=torch.argmax(output)
        return output

    def final_test(self, X_test, y_test):
        score=0
        for i in range(len(X_test)):
            predict = self.predict(torch.tensor(X_test[i]))
            if predict == y_test[i]:
                score += 1
        print('test score :', score/len(y_test))

    def cal_loss(self, x, y):
        result=self.forward(x, True)
        loss=-torch.log(result)[y]
        return loss

    def sdg_step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss =self.cal_loss(torch.tensor(x), torch.tensor(y))
        loss.backward()
        optimizer.step()

        return loss

    def train_step(self, X_train, y_train, X_dev, y_dev, nepoch=10):
        optimizer = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.01)
        num_examples_seen =0
        for epoch in range(nepoch):
            Loss=0
            # For each training example...
            for i in range(len(X_train)):
                # One SGD step
                Loss += self.sdg_step(X_train[i], y_train[i], optimizer).item()
                num_examples_seen += 1

            ######################### val accuracy ##################################
            score=0
            for i in range(len(X_dev)):
                predict=self.predict(torch.tensor(X_dev[i]))
                if predict==y_dev[i]:
                    score+=1

            # model 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_examples_seen': num_examples_seen
            }, 'saved_model_epoch=' + str(epoch))


            print('Train loss : %08f' % float(Loss/len(X_train)) , '    validation accuracy :', score/len(y_dev))




if __name__ == '__main__':

    # generate_batch 안의 인자는 파일명
    X_train, y_train=generate_batch("project2_train.xlsx")
    X_dev, y_dev=generate_batch("project2_dev.xlsx")
    X_test, y_test = generate_batch("project2_test.xlsx")
    print('X_train :', X_train)
    print('y_train :', y_train)
    print('X_dev :', X_dev)
    print('y_dev :', y_dev)
    print('X_test :', X_dev)
    print('y_test :', y_dev)

    # model을 정의
    model=classification()

    print('\nStart training!!!')
    # model training
    model.train_step(X_train, y_train, X_dev, y_dev)
    print('\nfinal testing!!!')
    model.final_test(X_test, y_test)
