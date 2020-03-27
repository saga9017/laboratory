# f= open(r'C:\Users\Lee Wook Jin\PycharmProjects\project4\complete_crawling\img(0~98)\repost\repost_preprocessed.txt' , 'r', encoding='utf-8')
# f_w= open(r'C:\Users\Lee Wook Jin\PycharmProjects\project4\complete_crawling\img(0~98)\repost\repost_preprocessed2.txt' , 'w', encoding='utf-8')
# line=f.readline()
# f_w.write('\n')
# count=1
# while True:
#     line=f.readline()
#     if not line:break
#     print(line.split('\t'))
#     f_w.write('Post_'+str(count)+'\t'+line.split('\t')[1])
#     count+=1
# f.close()
# f_w.close()


f= open(r'C:\Users\Lee Wook Jin\PycharmProjects\project4\complete_crawling\img(0~98)\repost\repost_loc.txt' , 'r', encoding='utf-8')
f_w= open(r'C:\Users\Lee Wook Jin\PycharmProjects\project4\complete_crawling\img(0~98)\repost\repost_loc2.txt' , 'w', encoding='utf-8')
count=1
while True:
    line=f.readline()
    if not line:break
    print(line.split('\t'))
    f_w.write('Post_'+str(count)+'\t'+line.split('\t')[1])
    count+=1
f.close()
f_w.close()