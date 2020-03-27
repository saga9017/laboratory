import tensorflow as tf
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # GPU number to use

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
from datetime import datetime
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

#######################################################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
##########################################################################################################################################
#######################################################################################################################################
df = pd.read_csv("in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df.sample(10)
# Create sentence and label lists
sentences = df.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])
##########################################################################################################################################
MAX_LEN = 128
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_inputs=input_ids
train_labels=labels
train_masks=attention_masks
##########################################################################################################################################
##########################################################################################################################################
df = pd.read_csv("in_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
df.shape
df.sample(10)
# Create sentence and label lists
sentences = df.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])
##########################################################################################################################################
MAX_LEN = 128
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

validation_inputs=input_ids
validation_labels=labels
validation_masks=attention_masks

#train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, random_state=2018, test_size=0.1)
#train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,random_state=2018, test_size=0.1)
##########################################################################################################################################
##########################################################################################################################################
# Convert all of our data into torch tensors, the required datatype for our model

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32
batch_size = 32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_data
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2531)
model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]


# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def onehot(index_array: np.ndarray, nb_classes: int, dtype=np.float32):
    return np.eye(nb_classes, dtype=dtype)[index_array]


def confusion_per_class(predictions: np.ndarray, labels: np.ndarray, nb_classes: int):
    predictions = onehot(predictions, nb_classes, np.bool8)
    labels = onehot(labels, nb_classes, np.bool8)

    n_predictions = ~predictions
    n_labels = ~labels

    tp_per_class = (predictions & labels).sum(0).astype(np.float32)
    fp_per_class = (predictions & n_labels).sum(0).astype(np.float32)
    fn_per_class = (n_predictions & labels).sum(0).astype(np.float32)
    return tp_per_class, fp_per_class, fn_per_class


def micro_f1_score(tp_per_class, fp_per_class, fn_per_class):
    total_tp = tp_per_class.sum()
    total_fp = fp_per_class.sum()
    total_fn = fn_per_class.sum()
    del tp_per_class
    del fp_per_class
    del fn_per_class

    total_precision = total_tp / (total_tp + total_fp + 1e-12)
    total_recall = total_tp / (total_tp + total_fn + 1e-12)

    micro_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-12)
    del total_precision
    del total_recall

    return micro_f1


def www_macro_f1_score(tp_per_class: np.ndarray, fp_per_class: np.ndarray, fn_per_class: np.ndarray):
    is_nonzero_prediction = (tp_per_class + fp_per_class) != 0
    is_nonzero_actual = (tp_per_class + fn_per_class) != 0

    where_nonzero_prediction = np.squeeze(is_nonzero_prediction.nonzero(), axis=-1)
    where_nonzero_actual = np.squeeze(is_nonzero_actual.nonzero(), axis=-1)
    del is_nonzero_prediction, is_nonzero_actual

    precision_per_class = tp_per_class[where_nonzero_prediction] / (
            tp_per_class[where_nonzero_prediction] + fp_per_class[where_nonzero_prediction])
    recall_per_class = tp_per_class[where_nonzero_actual] / (
                tp_per_class[where_nonzero_actual] + fn_per_class[where_nonzero_actual])
    del tp_per_class, fp_per_class, fn_per_class

    macro_precision = precision_per_class.mean()
    macro_recall = recall_per_class.mean()
    del precision_per_class, recall_per_class
    if macro_precision + macro_recall == 0:
        macro_f1 = 0
    else:
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
    return macro_f1


def compute_metrics(preds, labels, nb_classes):
    assert len(preds) == len(labels)

    tp_per_class, fp_per_class, fn_per_class = confusion_per_class(preds, labels, nb_classes)
    micro_f1 = micro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    macro_f1 = www_macro_f1_score(tp_per_class, fp_per_class, fn_per_class)
    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 20

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):

    # Training

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()

        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        if nb_tr_steps % 100 == 0:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(time, ' ', int(100 * nb_tr_steps / len(train_dataloader)), end='')
            print('%   완료!!!', end='')
            print('   loss :', tr_loss / nb_tr_steps)

    print("Train loss: {}".format(tr_loss / nb_tr_steps))

    # Validation

    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    micro_f1, macro_f1 = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        tmp_f1 = compute_metrics(np.argmax(logits, 1), label_ids, 2531)

        eval_accuracy += tmp_eval_accuracy
        micro_f1 += tmp_f1['micro_f1']
        macro_f1 += tmp_f1['macro_f1']
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Micro-f1: {}".format(micro_f1 / nb_eval_steps))
    print("Macro-f1: {}".format(macro_f1 / nb_eval_steps))