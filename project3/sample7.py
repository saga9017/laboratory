import torch
import torch.nn as nn
from transformers import *

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


softmax=nn.Softmax(-1)
citerion = nn.CrossEntropyLoss()
a=softmax(torch.randn(5,20))
b=torch.tensor([1,2,3,4,5])

print(a)
print(b)

loss=citerion(a,b)

print(loss)

x='I like the woman .'
print(tokenizer.encode(x))
print(tokenizer.cls_token_id)