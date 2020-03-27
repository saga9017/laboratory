from transformer_model import transformer
from preprocessing import data
import torch

X_train, y_train, label_number= data()

model = transformer(label_number)
model.load_state_dict(torch.load('saved_model'))

print(model.predict(X_train[:3]))