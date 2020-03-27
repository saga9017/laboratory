import numpy as np
import operator

input=[9,10]
output=[1,2,3,4,5,6,7]
x=[6,4,2,8,9,1]
vocab=['a','c','d','h','f','g']

dic_prob={'a1':0.1 ,'b1':0.2, 'c1': 0.3, 'd1': 0.4}
#dic_prob[('c', 'd')]=1


print(len(dic_prob.keys()))


def lowest_prob_pair(p):
    assert (len(p) >= 2)  # Ensure there are at least 2 symbols in the dist.
    sorted_p= sorted(p.items(), key=operator.itemgetter(1))
    return sorted_p[0], sorted_p[1]


class Node(object):
    def __init__(self, data):
        self.data = data
        self.index=None
        self.prob=None
        self.parent=None
        self.left = self.right=None



def preorder(self):
    print("!!!!", self.index, end=" ")
    if self.left:
        preorder(self.left)
    if self.right:
        preorder(self.right)

def parent(self):
    if self.parent==None:
        return None
    print("!!!!", self.parent.index, end=" ")
    parent(self.parent)

def isString(s):
  try:
    str(s)
    return True
  except ValueError:
    return False



#huffman code의 생성
huffman_code=dic_prob.copy()

nodes={}
node_index=-1


while len(huffman_code.keys())!=1:
    a1, a2 = lowest_prob_pair(huffman_code)
    huffman_code[(a1[0], a2[0])] = a1[1] + a2[1]
    del huffman_code[a1[0]]
    del huffman_code[a2[0]]
    nodes[(a1[0], a2[0])]=Node((a1[0], a2[0]))
    node=nodes[(a1[0], a2[0])]
    node_index += 1
    node.index = node_index
    node.prob=a1[1]/(a1[1]+a2[1])

    if a1[0] in nodes.keys():
        node.left=nodes[a1[0]]
        nodes[a1[0]].parent=node
    else:
        nodes[a1[0]]=Node(a1[0])
        node.left = nodes[a1[0]]
        nodes[a1[0]].parent = node

    if a2[0] in nodes.keys():
        node.right = nodes[a2[0]]
        nodes[a2[0]].parent = node
    else:
        nodes[a2[0]] = Node(a2[0])
        node.right = nodes[a2[0]]
        nodes[a2[0]].parent = node



root=list(huffman_code.keys())[0]
root_node=nodes[root]
print('root :', root)
print('root_node :', root_node)
huffman_code[root]=''


pivot=root
print(type(pivot)==tuple)
return_pivots=[]
while len(huffman_code.keys())<len(dic_prob.keys()):
    print(len(huffman_code.keys()))
    huffman_code[pivot[0]]=huffman_code[pivot]+'0'
    huffman_code[pivot[1]]=huffman_code[pivot]+'1'

    del huffman_code[pivot]

    if type(pivot[1]) == tuple:
        return_pivots.insert(0, pivot[1])

    if type(pivot[0])==tuple:
        pivot=pivot[0]

    else:
        if return_pivots==[]:
            pivot=None
        else:
            pivot=return_pivots.pop()



"""""
node_index=-1

#huffman_tree의 구현
def huffman_tree(root):
    if root in dic_prob.keys():
        node = Node(root)
        node.left = None
        node.right = None
        return node
    left=root[0]
    right=root[1]
    node=Node(root)
    global node_index
    node_index += 1
    node.index=node_index
    node.left=huffman_tree(left)
    node.right=huffman_tree(right)
    return node
"""""

print('dic_prob :', dic_prob)


preorder(root_node)
print()
print('node_index :',node_index)

print('huffman_code :', huffman_code)
print("활성 node :", huffman_code['a1'])


parent(nodes['c1'])


"""""
for k in range(-2, 2+1):
 if k != 0:
     print(k)

for j, unigram in enumerate(vocab):
 print(j, unigram)
"""""

index_of_word={'abcde': 1, 'abcgh': 2, 'u':3}
print(x[-2:])

n_gram=3
n_gram_dic={}
n_gram_number=0
for x in index_of_word.keys():
    y='<'+x+'>'
    for i  in range(len(y)-n_gram+1):
        if y[i:i+n_gram] not in n_gram_dic.keys():
            n_gram_dic[y[i:i+n_gram]]=n_gram_number
            n_gram_number += 1



print(n_gram_dic, n_gram_number)
#print(np.log(output))
#print(np.argsort(x))
#print("a"+"b")





