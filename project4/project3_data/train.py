import sys, os
import numpy as np
import random
import copy
import time
from project3_data.result_calculator import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import pickle

import tensorflow as tf
from project3_data.config import *

if evaluation_factor() == '1factor':
    from model_1factor import model_build
elif evaluation_factor() == '2factors':
    from model_2factors import model_build
elif evaluation_factor() == '3factors':
    from model_3factors_tlk import model_build
else:  # 4factors
    from model_4factors import model_build


# ★ oversampling
# def train_generate_batch(t_maxlen, l_maxlen, hashtag_size, hashtagCount, hashtagCount_saved, tag_used):
#     batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
#     batch_text = np.zeros((batch_size(), t_maxlen))
#     batch_loc = np.zeros((batch_size(), l_maxlen))
#     batch_y = np.zeros(batch_size())
#     truth_hashtag = []
#     text_start = []
#     epoch_finish = False
#     for i in range(batch_size()):
#
#         hashtag_choice = random.randrange(0, hashtag_size)
#         data_choice = random.randrange(0, len(hashtagCount[hashtag_choice]))
#         # print(hashtag_choice, data_choice, "/", len(hashtagCount[hashtag_choice]))
#         # while True:
#         #     hashtag_choice = random.randrange(0, hashtag_size)
#         #     if tag_used[hashtag_choice] == False:
#         #         data_choice = random.randrange(0, len(hashtagCount[hashtag_choice]))
#         #         break
#
#         data_index = hashtagCount[hashtag_choice][data_choice]
#         batch_img[i] = train_data[0][data_index]
#         batch_text[i] = train_data[1][data_index]
#         batch_loc[i] = train_data[2][data_index]
#         batch_y[i] = hashtag_choice
#         truth_hashtag.append(train_data[3][data_index])
#
#         allzero = False
#         for q, j in enumerate(batch_text[i]):
#             if int(j) != 0:
#                 text_start.append(q)
#                 allzero = True
#                 break
#         if allzero == False: text_start.append(0)
#
#         del hashtagCount[hashtag_choice][data_choice]
#         if len(hashtagCount[hashtag_choice]) == 0:
#             tag_used[hashtag_choice] = True
#             hashtagCount[hashtag_choice] = copy.deepcopy(hashtagCount_saved[hashtag_choice])
#             if np.all(tag_used) == True:
#                 print("다썼다!")
#                 tag_used = [False for g in range(hashtag_size)]
#                 epoch_finish = True
#                 break
#
#     return batch_img, batch_text, batch_loc, batch_y, epoch_finish, truth_hashtag, text_start, tag_used, hashtagCount

# ★ shuffle batch
# def generate_batch(which, pnt, y_cnt, t_maxlen, l_maxlen, finish):
#     batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
#     batch_text = np.zeros((batch_size(), t_maxlen))
#     batch_loc = np.zeros((batch_size(), l_maxlen))
#     batch_y = np.zeros(batch_size())
#     batch_cnt = 0
#     truth_hashtag = []
#     shuffle = list(range(batch_size()))
#     random.shuffle(shuffle)
#     while True:
#         text_start = []
#         if which == "train":
#             hashend = len(train_data[3][pnt])
#             datalen = len(train_data[0])
#             batch_img[shuffle[batch_cnt]] = train_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = train_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = train_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = train_data[3][pnt][y_cnt]
#             truth_hashtag.append(train_data[3][pnt])
#         elif which == "validation":
#             hashend = len(val_data[3][pnt])
#             datalen = len(val_data[0])
#             batch_img[shuffle[batch_cnt]] = val_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = val_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = val_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = val_data[3][pnt][y_cnt]
#             truth_hashtag.append(val_data[3][pnt])
#         else:
#             hashend = len(test_data[3][pnt])
#             datalen = len(test_data[0])
#             batch_img[shuffle[batch_cnt]] = test_data[0][pnt]
#             batch_text[shuffle[batch_cnt]] = test_data[1][pnt]
#             batch_loc[shuffle[batch_cnt]] = test_data[2][pnt]
#             batch_y[shuffle[batch_cnt]] = test_data[3][pnt][y_cnt]
#             truth_hashtag.append(test_data[3][pnt])
#
#         allzero = False
#         for i, j in enumerate(batch_text[shuffle[batch_cnt]]):
#             if int(j) != 0:
#                 text_start.append(i)
#                 allzero = True
#                 break
#         if allzero == False: text_start.append(0)
#
#         # print("------------------------------------------")
#         # print("input text:")
#         # for i in batch_text[batch_cnt]:
#         #     textnum = int(i)
#         #     if textnum != 0:
#         #         print(vocabulary_inv[textnum], end=" ")
#         # print("\ninput loc:")
#         # for i in batch_loc[batch_cnt]:
#         #     locnum = int(i)
#         #     if locnum != 0:
#         #         print(vocabulary_inv[locnum], end=" ")
#         # print("\nTrue hashtag:")
#         # for i in truth_hashtag[batch_cnt]:
#         #     print(hashtagVoc_inv[int(i)], end="||")
#         # print()
#         y_cnt += 1
#         batch_cnt += 1
#
#         if y_cnt == hashend:
#             y_cnt = 0
#             pnt += 1
#             if pnt == datalen:
#                 pnt = 0
#                 finish = True
#
#         if finish or batch_cnt == batch_size(): break
#     return batch_img, batch_text, batch_loc, batch_y, pnt, y_cnt, finish, truth_hashtag, text_start


def load_k():
    k_train, k_val, k_test = [], [], []
    with open("./data/txt/e5_train_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_train.append(line)

    with open("./data/txt/e5_val_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_val.append(line)

    with open("./data/txt/e5_test_insta_textonly.txt", "r") as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            for i in range(len(line)):
                line[i] = int(line[i])
            k_test.append(line)

    return k_train, k_val, k_test


def generate_batch(which, pnt, y_cnt, t_maxlen, l_maxlen, finish):
    batch_img = np.zeros((batch_size(), 1, feat_dim(), w(), w()))
    batch_text = np.zeros((batch_size(), t_maxlen))
    batch_loc = np.zeros((batch_size(), l_maxlen))
    batch_know = np.zeros((batch_size(), cat_num()))
    batch_y = np.zeros(batch_size())
    batch_cnt = 0
    truth_hashtag = []

    while True:
        if which == "train":
            hashend = len(train_data[3][pnt])
            datalen = len(train_data[0])
            batch_img[batch_cnt] = train_data[0][pnt]
            batch_text[batch_cnt] = train_data[1][pnt]
            batch_loc[batch_cnt] = train_data[2][pnt]
            batch_y[batch_cnt] = train_data[3][pnt][y_cnt]
            truth_hashtag.append(train_data[3][pnt])
            batch_know[batch_cnt] = train_data[4][pnt]
        elif which == "validation":
            hashend = len(val_data[3][pnt])
            datalen = len(val_data[0])
            batch_img[batch_cnt] = val_data[0][pnt]
            batch_text[batch_cnt] = val_data[1][pnt]
            batch_loc[batch_cnt] = val_data[2][pnt]
            batch_y[batch_cnt] = val_data[3][pnt][y_cnt]
            truth_hashtag.append(val_data[3][pnt])
            batch_know[batch_cnt] = val_data[4][pnt]
        else:
            hashend = len(test_data[3][pnt])
            datalen = len(test_data[0])
            batch_img[batch_cnt] = test_data[0][pnt]
            batch_text[batch_cnt] = test_data[1][pnt]
            batch_loc[batch_cnt] = test_data[2][pnt]
            batch_y[batch_cnt] = test_data[3][pnt][y_cnt]
            truth_hashtag.append(test_data[3][pnt])
            batch_know[batch_cnt] = test_data[4][pnt]

        y_cnt += 1
        batch_cnt += 1

        if y_cnt == hashend:
            y_cnt = 0
            pnt += 1
            if pnt == datalen:
                pnt = 0
                finish = True

        if finish or batch_cnt == batch_size(): break
    return batch_img, batch_text, batch_loc, batch_know, batch_y, pnt, y_cnt, finish, truth_hashtag


if __name__ == '__main__':
    print("co-attention_" + evaluation_factor())
    if evaluation_factor() == '1factor':
        print("factor:", which_factor())
    if evaluation_factor() == '2factors' or evaluation_factor() == '3factors':
        print("connection:", connection())

    print("last function:", last_function())
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
    val_data.append(np.load("./data/val_img_x_c_h.npz")["img"])
    print("validation data loading finished.")
    with open("./data/val_tlh_keras_h.bin", "rb") as f:
        val_data.extend(pickle.load(f))
    print("validation data 업로드")
    print(len(val_data[0]), len(val_data[1]), len(val_data[2]), len(val_data[3]))
    # val_data = check_hashzero(val_data)
    # print("check 완")

    test_data = []
    test_data.append(np.load("./data/test_img_x_c_h.npz")["img"])
    print("test data loading finished.")
    with open("./data/test_tlh_keras_h.bin", "rb") as f:
        test_data.extend(pickle.load(f))
    print("test data 업로드")
    print(len(test_data[0]), len(test_data[1]), len(test_data[2]), len(test_data[3]))
    # test_data = check_hashzero(test_data)
    # print("check 완")

    train_data = []
    train_data.append(np.load("./data/train_img_x_c_h.npz")["img"])
    print("train data loading finished.")
    with open("./data/train_tlh_keras_h.bin", "rb") as f:
        train_data.extend(pickle.load(f))
    print("train data 업로드")
    print(len(train_data[0]), len(train_data[1]), len(train_data[2]), len(train_data[3]))

    print("train data size: ", len(train_data[3]))

    text_maxlen = len(train_data[1][0])  # 411 맞는지 확인
    loc_maxlen = len(train_data[2][0])  # 411
    print("text_maxlen:", text_maxlen)
    print("loc_maxlen:", loc_maxlen)
    vocab_size = len(vocabulary_inv)  # 26210
    hashtag_size = len(hashtagVoc_inv)  # 2988
    print('-')
    print('Vocab size:', vocab_size, 'unique words')
    print('Hashtag size:', hashtag_size, 'unique hashtag')

    for index, taglist in enumerate(train_data[3]):
        for tag in taglist:
            hashtagCount[int(tag)].append(index)

    cnt = 0
    for i in list(hashtagCount.keys()):
        if len(hashtagCount[i]) == 0:
            del hashtagCount[i]

    hashtagCount_saved = copy.deepcopy(hashtagCount)
    model = model_build(vocab_size, text_maxlen, hashtag_size)

    # Knowledge-base 추가
    k_train, k_val, k_test = load_k()
    print(len(k_train), len(k_val), len(k_test))
    train_data.append(k_train)
    val_data.append(k_val)
    test_data.append(k_test)

    print("starts training...")
    best_f1 = 0
    train_loss = []
    val_loss = []
    val_precision = []
    val_recall = []
    val_f1 = []
    prev_pre = 0

    test_epoch = []
    test_precision = []
    test_recall = []
    test_f1 = []

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

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
                batch_img, batch_text, batch_loc, batch_know, batch_y, train_pnt, y_cnt, epoch_finish, truth_hashtag = \
                    generate_batch("train", train_pnt, y_cnt, text_maxlen, loc_maxlen, epoch_finish)
                # print(np.shape(batch_know)) # [batch_size, cat_num]

                if epoch_finish: break
                feed_dict = {model.textplaceholder: batch_text,
                             model.imgplaceholder: batch_img,
                             model.locplaceholder: batch_loc,
                             model.knowplaceholder: batch_know,
                             model.yplaceholder: batch_y,
                             model.keep_prob: 0.5}
                if evaluation_factor() == '1factor':

                    if which_factor() == "i":
                        _, c = sess.run([model.i_optimizer, model.i_cost],
                                        feed_dict=feed_dict)
                    elif which_factor() == "t":
                        _, c = sess.run([model.t_optimizer, model.t_cost],
                                        feed_dict=feed_dict)
                    elif which_factor() == "l":
                        _, c = sess.run([model.l_optimizer, model.l_cost],
                                        feed_dict=feed_dict)
                    else:  # k
                        _, c = sess.run([model.k_optimizer, model.k_cost],
                                        feed_dict=feed_dict)

                # it, il, lt, ik, tk, lk, itl, itk, ilk, tlk,imgatt
                elif evaluation_factor() == '2factors':
                    if connection() == "it":
                        _, c = sess.run([model.it_optimizer, model.it_cost],
                                        feed_dict=feed_dict)
                    elif connection() == "il":
                        _, c = sess.run([model.il_optimizer, model.il_cost],
                                        feed_dict=feed_dict)
                    elif connection() == "lt":
                        _, c = sess.run([model.lt_optimizer, model.lt_cost],
                                        feed_dict=feed_dict)
                    elif connection() == "ik":
                        _, c = sess.run([model.ik_optimizer, model.ik_cost],
                                        feed_dict=feed_dict)
                    elif connection() == "tk":
                        _, c = sess.run([model.tk_optimizer, model.tk_cost],
                                        feed_dict=feed_dict)
                    elif connection() == "lk":  # lk
                        _, c = sess.run([model.lk_optimizer, model.lk_cost],
                                        feed_dict=feed_dict)
                    else:   # imgatt
                        _, c = sess.run([model.imgatt_it_optimizer, model.imgatt_it_cost],
                                        feed_dict=feed_dict)
                else:  # 3factors or 4factors
                    _, c = sess.run([model.optimizer, model.cost],
                                    feed_dict=feed_dict)

                train_epoch_loss += c
                batchloss += c

            print('Epoch', epoch + 1, 'completed out of', nb_epoch(), 'loss:', train_epoch_loss)
            train_loss.append(train_epoch_loss)
            # valid set : 8104개
            val_pnt = 0
            y_cnt = 0
            val_pred = []
            val_truth = []
            val_epoch_loss = 0

            while True:
                epoch_finish = False
                batch_img, batch_text, batch_loc, batch_know, batch_y, val_pnt, y_cnt, epoch_finish, truth_hashtag = \
                    generate_batch("validation", val_pnt, y_cnt, text_maxlen, loc_maxlen, epoch_finish)
                if epoch_finish: break
                feed_dict = {model.textplaceholder: batch_text,
                             model.imgplaceholder: batch_img,
                             model.locplaceholder: batch_loc,
                             model.knowplaceholder: batch_know,
                             model.yplaceholder: batch_y,
                             model.keep_prob: 1}
                if evaluation_factor() == '1factor':
                    if which_factor() == "i":
                        val_prob, c = sess.run([model.i_final_output, model.i_cost],
                                               feed_dict=feed_dict)
                    elif which_factor() == "t":
                        val_prob, c = sess.run([model.t_final_output, model.t_cost],
                                               feed_dict=feed_dict)
                    elif which_factor() == "l":
                        val_prob, c = sess.run([model.l_final_output, model.l_cost],
                                               feed_dict=feed_dict)
                    else:  # k
                        val_prob, c = sess.run([model.k_final_output, model.k_cost],
                                               feed_dict=feed_dict)
                elif evaluation_factor() == '2factors':
                    if connection() == "it":
                        val_prob, c = sess.run([model.it_final_output, model.it_cost],
                                               feed_dict=feed_dict)
                    elif connection() == "il":
                        val_prob, c = sess.run([model.il_final_output, model.il_cost],
                                               feed_dict=feed_dict)
                    elif connection() == "lt":
                        val_prob, c = sess.run([model.lt_final_output, model.lt_cost],
                                               feed_dict=feed_dict)
                    elif connection() == "ik":
                        val_prob, c = sess.run([model.ik_final_output, model.ik_cost],
                                               feed_dict=feed_dict)
                    elif connection() == "tk":
                        val_prob, c = sess.run([model.tk_final_output, model.tk_cost],
                                               feed_dict=feed_dict)
                    elif connection() == "lk":
                        val_prob, c = sess.run([model.lk_final_output, model.lk_cost],
                                               feed_dict=feed_dict)
                    else:    # imgatt
                        val_prob, c = sess.run([model.imgatt_it_final_output, model.imgatt_it_cost],
                                               feed_dict=feed_dict)

                else:  # 3factors or 4factors
                    val_prob, c = sess.run([model.final_output, model.cost],
                                           feed_dict=feed_dict)

                val_epoch_loss += c
                # if last_function() == "softmax":
                y_pred = np.argsort(val_prob, axis=1)
                for i in range(batch_size()):
                    val_pred.append(y_pred[i])
                    val_truth.append(truth_hashtag[i])

            precision = top1_acc(val_truth, val_pred)

            print("Epoch:", (epoch + 1), "validation loss:", val_epoch_loss, "val_precision:", precision)
            val_loss.append(val_epoch_loss)
            val_precision.append(precision)

            if precision > prev_pre:
                if evaluation_factor() == '1factor':
                    saver.save(sess, './saved_model/' + evaluation_factor() + '_' + which_factor() + '/'
                               + evaluation_factor() + '_' + which_factor(),
                               global_step=epoch + 1)
                elif evaluation_factor() == '2factors' or evaluation_factor() == '3factors':
                    saver.save(sess, './saved_model/' + evaluation_factor() + '_' + connection() + '/'
                               + evaluation_factor() + '_' + connection(),
                               global_step=epoch + 1)
                else:  # 4factors
                    saver.save(sess, './saved_model/' + evaluation_factor() + '/'
                               + evaluation_factor(), global_step=epoch + 1)

                prev_pre = precision
                max_epoch = epoch + 1
                # attention_weights = sess.run(model.attention_weights,
                #                              feed_dict=feed_dict)

    print("\ntop1 결과 정리")
    print("loss comparing")
    for i in range(len(train_loss)):
        print("epoch", i, "| train loss:", train_loss[i], ", validation loss:", val_loss[i])
    print("validation")
    for i in range(len(val_precision)):
        print("epoch", i + 1, "- precision:", val_precision[i])
    # print("attention vector weights")
    # print_order = ["i_w_it", "i_w_il", "i_w_ik", "t_w_it", "t_w_lt", "t_w_tk", "l_w_lt",
    #                "l_w_il", "l_w_lk", "k_w_ik", "k_w_tk", "k_w_lk"]
    # print(attention_weights)
    # with open("./result_weight/" + evaluation_factor() + ".txt", "w") as ff:
    #     ff.write("순서대로 i_w_it, i_w_il, i_w_ik, t_w_it, t_w_lt, t_w_tk, l_w_lt, l_w_il, l_w_lk, k_w_ik, k_w_tk, k_w_lk")
    #     for q in range(len(attention_weights)):
    #         ff.write(print_order[q] + " : " + str(attention_weights[q]) + "\n")
    # with open("./result_weight/" + evaluation_factor() + ".bin", "wb") as ff:
    #     pickle.dump(attention_weights, ff)
    #
