import sqlite3
import numpy as np
import io
import time

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sqlite3.register_adapter(np.ndarray, adapt_array)

# Converts TEXT to np.array when selecting
sqlite3.register_converter("array", convert_array)




con = sqlite3.connect("./test.db", detect_types=sqlite3.PARSE_DECLTYPES)
cur = con.cursor()


"""""
b1 = np.arange(12).reshape(2,6)
b2 = np.arange(24).reshape(2,12)
b3 = np.arange(36).reshape(3,12)

#cur.execute("create table test (line integer, arr array)")


#cur.execute("insert into test values (?,?)", (1, b1))
#cur.execute("insert into test values (?,?)", (2, b2))
#cur.execute("insert into test values (?,?)", (3, b3))
#con.commit()
"""""
list1=[10000,2,3000,4,5000,6,7000,8,9000,10,11000,12,1300,14,15000,16,17000,18,19000,20,2100,22,23000,24,2500,26,270,28,29000,30,31,3200]
print(list1)
start=time.time()
cur.execute("select * from test where line in"+"{}".format(tuple(list1)))
data = dict(cur.fetchall())
print(time.time()-start)
print('============================')
print(data)
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
print(type(data))
# <type 'numpy.ndarray'>

"""""
a1=np.load("project3_transformer_data/bert_text_train4.npy")
total_step=len(a1)
for index, i in enumerate(a1):
    x=i
    cur.execute("insert into test values (?, ?)", (index+60000, x))
    if index % 200==0:
        print((index*100)/total_step, "% 완료")
        con.commit()
con.commit()
print(index+60000)


cur.execute("select * from test")
data = cur.fetchone()

print(data)
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]]
print(type(data))
# <type 'numpy.ndarray'>
"""""
print('finish!!!')