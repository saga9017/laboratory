import sys, os
from datetime import datetime
import numpy as np
import random
import copy
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
from transformers import *

from project3_data.result_calculator import *
import torch.nn.utils as torch_utils

pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)


bert = BertModel.from_pretrained(pretrained_weights,
                                 output_hidden_states=True,
                                 output_attentions=True, force_download=True)


class Bert(nn.Module):
    def __init__(self):
        # Assign instance variables
        super(Bert, self).__init__()
        self.bert = copy.deepcopy(bert)

    def forward(self, x, attention_mask, token_type_ids):

        outputs= self.bert(x, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_encoder_output = outputs[0]
        return last_encoder_output


