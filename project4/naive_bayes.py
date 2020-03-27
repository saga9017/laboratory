import sys, os
import numpy as np
import random
import copy
import time
import pickle
from result_calculator import *
from config import *

if __name__ == '__main__':
    # print("Naive-bayes")
    # print('loading data...')
    # with open("./data/vocabulary_keras_h.pkl", "rb") as f:
    #     data = pickle.load(f)
    # vocabulary = data[0]
    # hashtagVoc = data[2]
    # vocabulary_inv = {}
    # hashtagVoc_inv = {}
    # hashtagCount = {}
    # vocabulary["<novocab>"] = 0
    # for i in vocabulary.keys():
    #     vocabulary_inv[vocabulary[i]] = i
    # for i in hashtagVoc.keys():
    #     hashtagVoc_inv[hashtagVoc[i]] = i
    #     hashtagCount[hashtagVoc[i]] = []
    # print("vocabulary 스펙:", len(vocabulary), max(vocabulary.values()), min(vocabulary.values()))
    # print("hashtagVoc 스펙 :", len(hashtagVoc), max(hashtagVoc.values()), min(hashtagVoc.values()))
    # print("len(hashtagVoc_inv)", len(hashtagVoc_inv))
    #
    # with open("./data/train_data.pkl", "rb") as f:
    #     train_data = pickle.load(f)
    #
    # with open("./data/test_data.pkl", "rb") as f:
    #     test_data = pickle.load(f)
    #
    # print("train data size: ", len(train_data[3]))
    #
    # text_maxlen = len(train_data[1][0])  # 411 맞는지 확인
    # loc_maxlen = len(train_data[2][0])  # 411
    # print("text_maxlen:", text_maxlen)
    # print("loc_maxlen:", loc_maxlen)
    # vocab_size = len(vocabulary_inv)  # 26210
    # hashtag_size = len(hashtagVoc_inv)  # 2988
    # print('-')
    # print('Vocab size:', vocab_size, 'unique words')
    # print('Hashtag size:', hashtag_size, 'unique hashtag')
    #
    # for index, taglist in enumerate(train_data[3]):
    #     for tag in taglist:
    #         hashtagCount[int(tag)].append(index)
    #
    # cnt = 0
    # for i in list(hashtagCount.keys()):
    #     if len(hashtagCount[i]) == 0:
    #         del hashtagCount[i]
    #
    # hashtagCount_saved = copy.deepcopy(hashtagCount)
    # print("train")
    # print(len(train_data[0]), len(train_data[1]), len(train_data[2]), len(train_data[3]), len(train_data[4]))
    # print("test")
    # print(len(test_data[0]), len(test_data[1]), len(test_data[2]), len(test_data[3]), len(test_data[4]))
    #
    # t_train = []
    # for i in range(len(train_data[1])):
    #     if i % 10000 == 0:
    #         print("진행률:", i / len(train_data[1]) * 100)
    #     t_train.append([])
    #     for j in range(len(train_data[1][i])):
    #         if train_data[1][i][j] != 0:
    #             t_train[i].append(train_data[1][i][j])
    #
    # t_test = []
    # for i in range(len(test_data[1])):
    #     t_test.append([])
    #     for j in range(len(test_data[1][i])):
    #         if test_data[1][i][j] != 0:
    #             t_test[i].append(test_data[1][i][j])
    #

    word_vocab={}

    hashtag_size = 2953
    vocab_size=26405
    prior = np.zeros([hashtag_size])
    words_count = np.zeros([hashtag_size, vocab_size]) + 0.0000001

    with open('data3.pickle', 'rb') as f:
        data = pickle.load(f)

    print(len(data))

    val_data_text = data[:2227]
    train_data_text = data[4454:]

    val_data = []

    new_val_loc_text_list = []
    new_val_loc_list = []
    val_label = []
    for idx, i in enumerate(val_data_text):
        tmp=[]
        for j in i[0]:
            if j not in word_vocab:
                word_vocab[j]=len(word_vocab)
            tmp.append(word_vocab[j])
        new_val_loc_text_list.append(tmp)
        val_label.append(i[1])
    val_data.append(new_val_loc_text_list)
    val_data.append(val_label)


    print(len(val_data[0]))
    # val_data = check_hashzero(val_data)
    # print("check 완")


    train_data = []


    new_train_loc_text_list = []
    new_train_loc_list = []
    train_label = []
    for idx, i in enumerate(train_data_text):
        tmp = []
        for j in i[0]:
            if j not in word_vocab:
                word_vocab[j] = len(word_vocab)
            tmp.append(word_vocab[j])
        new_train_loc_text_list.append(tmp)
        train_label.append(i[1])

    train_data.append(new_train_loc_text_list)
    train_data.append(train_label)

    print(len(train_data[0]))





    for i in range(len(new_train_loc_text_list)):
        if i % 1000 == 0:
            print("진행률:", i / len(train_data[1]) * 100)
        for hashtagclass in train_data[1][i]:
            for word in new_train_loc_text_list[i]:
                words_count[hashtagclass][word] += 1
            prior[hashtagclass] += 1
    prior /= np.sum(prior)
    prior += 0.0000001
    class_count = np.sum(words_count, axis=0)
    acc = 0
    test_precision = []
    test_recall = []
    for i in range(len(new_val_loc_text_list)):
        if i % 500 == 0:
            print("test진행률:", i / len(new_val_loc_text_list) * 100)
        test_precision.append([])
        test_recall.append([])

        test_prob = np.zeros([hashtag_size, len(new_val_loc_text_list[i])])
        for j in range(len(new_val_loc_text_list[i])):
            for k in range(hashtag_size):
                test_prob[k][j] = words_count[k][new_val_loc_text_list[i][j]]
        test_prob = np.prod(test_prob, axis=1)
        final_array = np.flip(np.argsort(np.multiply(test_prob, prior)))
        final = final_array[0]
        answer = []
        match = "X"
        for k in val_data[1][i]:
            answer.append(k)
        if final in answer:
            acc += 1
            match = "O"
        # print(answer, hashtagVoc_inv[final], match)
        test_precision[i], test_recall[i] = pr_score(val_data[1][i], final_array, hashtag_size)

    acc /= len(new_val_loc_text_list)
    print("top-1 accuracy : ", acc)
    filename = "./result/naive_bayes.bin"
    with open(filename, "wb+") as f:
        pickle.dump([[], test_precision, test_recall], f)
