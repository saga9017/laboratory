How to run

python word2vec.py [mode] [negative_samples] [partition]

mode : "SG" for skipgram, "CBOW" for CBOW
negative_samples : 0 for hierarchical softmax, the other numbers will be the number of negative samples (20 recommended)
partition : "part" if you want to train on a part of corpus (faster training but worse performance), 
             "full" if you want to train on full corpus (better performance but slower training)

Examples) 
python word2vec.py SG 0 full // SG training with hierarchical softmax
python word2vec.py SG 20 full // SG training with 20 negative samples

You should adjust the other hyperparameters in the code file manually.

========================================================================================

https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf

In above paper, there are many techniques for boosting performance such as learning rate scheduling, hyper-parameter settings, ....

You can use any of them.