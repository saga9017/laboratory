f=open('new.txt', 'r')
while True:
    line=f.readline()
    if not line:break
    print(line.split('\t'))
f.close()