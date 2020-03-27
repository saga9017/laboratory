from .crawler import get_posts_by_hashtag
from .crawler import output

f = open("./data/top100cut", "r")
while True:
    line = f.readline()
    if not line:
        break
    tag = line[1:len(line) - 1]
    print(tag)
    outputFile = "./data/hashtag/" + tag
    output(get_posts_by_hashtag(tag, 100, action="store_true"), outputFile)
    print("Saved to ", outputFile)
