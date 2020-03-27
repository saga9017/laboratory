from transformer_model import transformer
from preprocessing import data, data_dev
import torch


X_train, y_train, label_number, label_dic= data()
X_dev, y_dev=data_dev()

# Train on a small subset of the data to see what happens
model = transformer(label_number+1)   #0부터 시작하므로
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


last_loss = model.train_with_batch(X_train[:5], y_train[:5], X_dev[:8], y_dev[:8], nepoch=300)

torch.save(model.state_dict(), 'saved_model')
print('model saved!!!')



