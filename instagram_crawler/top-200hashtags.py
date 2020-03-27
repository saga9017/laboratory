import pickle

f = open("./top-200 hashtags.txt", "r")
fw = open("./data/top200hashtags.txt", "w")
hashtags = list()
while True:
    line = f.readline()
    if not line: break
    if line.startswith("#"):
        print(line[:len(line)-1])
        hashtags.append(line[:len(line)-1])
        fw.write(line)
f.close()
fw.close()
print(hashtags)
# fw = open("./data/top200hashtags.txt", "w")
# fw.write(line)
# pickle.dump(hashtags, f)
# f.close()