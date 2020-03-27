import re

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

SST_2_train = []
SST_2_dev = []
SST_2_test = []
num_classes = 2  # the number of classes

# sentences of training data
with open('SST-2/stsa.binary.train', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_train.append([clean_str_sst(row[1:]), int(row[0])])

# sentences of validation data
with open('SST-2/stsa.binary.dev', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_dev.append([clean_str_sst(row[1:]), int(row[0])])

# sentences of test data
with open('SST-2/stsa.binary.test', 'r', encoding='latin-1') as f:
    for row in f.readlines():
        SST_2_test.append([clean_str_sst(row[1:]), int(row[0])])


print(SST_2_test)