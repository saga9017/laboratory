import sys, os
import numpy as np
import random
import copy
import time

import pickle



def generate_batch(which, pnt, y_cnt, maxlen, finish):
    batch_text = np.zeros((batch_size(), maxlen))
    batch_y = np.zeros(batch_size())
    batch_cnt = 0
    truth_hashtag = []

    while True:
        if which == "train":
            hashend = len(train_data[2][pnt])
            datalen = len(train_data[0])
            batch_text[batch_cnt] = train_data[0][pnt]
            batch_y[batch_cnt] = train_data[2][pnt][y_cnt]
            truth_hashtag.append(train_data[2][pnt])

        elif which == "validation":
            hashend = len(val_data[2][pnt])
            datalen = len(val_data[0])
            batch_text[batch_cnt] = val_data[0][pnt]
            batch_y[batch_cnt] = val_data[2][pnt][y_cnt]
            truth_hashtag.append(val_data[2][pnt])

        else:
            hashend = len(test_data[2][pnt])
            datalen = len(test_data[0])
            batch_text[batch_cnt] = test_data[0][pnt]
            batch_y[batch_cnt] = test_data[2][pnt][y_cnt]
            truth_hashtag.append(test_data[2][pnt])

        y_cnt += 1
        batch_cnt += 1
        if y_cnt == hashend:
            y_cnt = 0
            pnt += 1
            if pnt == datalen:
                pnt = 0
                finish = True

        print(batch_text)
        print(batch_y)
        print(train_pnt)
        print(y_cnt)
        print(epoch_finish)
        print(truth_hashtag)
        sys.exit(1)

        if finish or batch_cnt == batch_size(): break
    return batch_text, batch_y, pnt, y_cnt, finish, truth_hashtag


if __name__ == '__main__':
    print("current working directory:", os.getcwd())
    print('loading data...')

    with open("./data/vocabulary_keras_h.pkl", "rb") as f:
        data = pickle.load(f)
    vocabulary = data[0]
    hashtagVoc = data[2]
    vocabulary_inv = {}
    hashtagVoc_inv = {}
    hashtagCount = {}
    vocabulary["<novocab>"] = 0

    for i in vocabulary.keys():
        vocabulary_inv[vocabulary[i]] = i
    for i in hashtagVoc.keys():
        hashtagVoc_inv[hashtagVoc[i]] = i
        hashtagCount[hashtagVoc[i]] = []

    print("vocabulary 스펙:", len(vocabulary), max(vocabulary.values()), min(vocabulary.values()))
    print("hashtagVoc 스펙 :", len(hashtagVoc), max(hashtagVoc.values()), min(hashtagVoc.values()))
    print("len(hashtagVoc_inv)", len(hashtagVoc_inv))
    val_data = []
    with open("./data/val_tlh_keras_h.bin", "rb") as f:
        val_data.extend(pickle.load(f))
    print("validation data 업로드")
    print(len(val_data[0]), len(val_data[1]), len(val_data[2]))

    test_data = []
    with open("./data/test_tlh_keras_h.bin", "rb") as f:
        test_data.extend(pickle.load(f))
    print("test data 업로드")
    print(len(test_data[0]), len(test_data[1]), len(test_data[2]))

    train_data = []
    with open("./data/train_tlh_keras_h.bin", "rb") as f:
        train_data.extend(pickle.load(f))
    print("train data 업로드")
    print(len(train_data[0]), len(train_data[1]), len(train_data[2]))

    print("train data size: ", len(train_data[2]))

    text_maxlen = len(train_data[0][0])  # 411 맞는지 확인
    loc_maxlen = len(train_data[1][0])  # 411
    maxlen = text_maxlen
    print("text_maxlen:", text_maxlen)
    print("loc_maxlen:", loc_maxlen)
    vocab_size = len(vocabulary_inv)  # 26210
    hashtag_size = len(hashtagVoc_inv)  # 2988
    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Hashtag size:', hashtag_size, 'unique hashtag')

    for index, taglist in enumerate(train_data[2]):
        for tag in taglist:
            hashtagCount[int(tag)].append(index)

    cnt = 0
    for i in list(hashtagCount.keys()):
        if len(hashtagCount[i]) == 0:
            del hashtagCount[i]

    print("text data shape")
    print(train_data[0][1])
    cnt = 0
    print("-----------vocabulary-----------")
    print("<novocab>", vocabulary["<novocab>"])
    for i in vocabulary.keys():
        print(i, vocabulary[i])
        cnt += 1
        if cnt > 10: break

    cnt = 0
    print("-----------vocabulary_inv-----------")
    print("0", vocabulary_inv[0])
    for i in vocabulary_inv.keys():
        print(i, vocabulary_inv[i])
        cnt += 1
        if cnt > 10: break

    for epoch in range(nb_epoch()):
        print("에폭", epoch + 1)
        i = 0
        train_pnt = 0
        y_cnt = 0
        batchcount = 0
        batchloss = 0
        train_epoch_loss = 0
        tag_used = [False for g in range(hashtag_size)]

        while True:
            batchcount += 1
            if batchcount % 1000 == 0:
                print("배치,", batchcount, "번째, loss:", batchloss)
                batchloss = 0

            epoch_finish = False
            batch_text, batch_y, train_pnt, y_cnt, epoch_finish, truth_hashtag = \
                generate_batch("train", train_pnt, y_cnt, maxlen, epoch_finish)

            # print(np.shape(batch_know)) # [batch_size, cat_num]

            if epoch_finish: break

        print('Epoch', epoch + 1, 'completed out of', nb_epoch(), 'loss:', train_epoch_loss)

    print("\ntop1 결과 정리")
    print("loss comparing")
    for i in range(len(train_loss)):
        print("epoch", i, "| train loss:", train_loss[i], ", validation loss:", val_loss[i])
    print("validation")
    for i in range(len(val_precision)):
        print("epoch", i + 1, "- precision:", val_precision[i])
