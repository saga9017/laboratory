
def load_wordvector():
    f = open("saved_matrix/text8_skip-gram_negative-sampling", 'r')

    dic={}
    embeding_matrix=[]
    i=0
    while True:
        line = f.readline().split(' ')
        if not line[0]: break
        dic[line[0]]=i
        word_matrix= [float(s) for s in line[1:]]
        embeding_matrix.append(word_matrix)
        i+=1

    f.close()
    return dic, embeding_matrix