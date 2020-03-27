import copy
import numpy as np


def top1_acc(test_y, pred_y, k=1):
    p_score = 0
    for i in range(len(test_y)):
        result_at_top = pred_y[i][-k]
        if result_at_top in test_y[i]:
            p_score += 1
    return float(p_score / len(test_y))


def pr_score(test_truth, test_pred, hashtag_size):
    this_p = []
    this_r = []
    count = 0
    imsi = []
    test_truth = set(test_truth)
    for i in range(hashtag_size):
        if test_pred[i] in test_truth:
            count += 1
            imsi.append(test_pred[i])
        this_p.append(float(count / (i + 1)))
        this_r.append(float(count / len(test_truth)))
    if count != len(test_truth):
        print("False! count:", count, "/len(test_truth):", len(test_truth))
        print(test_truth)
        for i in range(len(test_pred)):
            print(test_pred[i], end="/")
        print()
    print("r 마지막:", this_r[-1], this_p[-1])
    return this_p, this_r


# def average_pr(test_truth, test_pred):
#     this_p = 0
#     test_truth = np.array(test_truth)
#     test_pred = np.array(test_pred)
#     where = []
#     for i in test_truth:
#         where.append(np.where(test_pred == i)[0])
#     where.sort()
#     for i in range(len(test_truth)):
#         this_p += (i+1) / (where[i]+1)
#     this_p /= len(where)
#     print(where)
#     print("this_p:",this_p)
#     return this_p


def precision_score(test_y, pred_y, count, k=1):
    p_score = []
    for i in range(len(test_y)):
        result_at_topk = pred_y[i][-k]
        if result_at_topk in test_y[i]:
            count[i] += 1
        p_score.append(float(count[i]) / float(k))

    return np.mean(p_score), count


def recall_score(test_y, pred_y, count, k=1):
    r_score = []
    for i in range(len(test_y)):
        result_at_topk = pred_y[i][-k]
        if result_at_topk in test_y[i]:
            count[i] += 1
        r_score.append(float(count[i]) / float(len(test_y[i])))

    return np.mean(r_score), count
