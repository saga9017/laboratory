def sum(x,y):
    result=[i+j for i,j in zip(x,y)]
    return result

def multiply(x, y):
    result=[i*y for i in x]
    return result

def subtraction(x, y):
    result = [i - j for i, j in zip(x, y)]
    return result