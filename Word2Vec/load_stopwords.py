
def load_stopwords():
    f = open("stopwords.txt", 'r')

    stopwords_list=[]

    while True:
        word = f.readline().split()
        if not word: break
        stopwords_list.append(word[0])

    f.close()
    return stopwords_list