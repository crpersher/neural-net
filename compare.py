import sys, os
sys.path.append(os.pardir)
import numpy as np

def compare(fileA = "generalization_test0.txt", fileB ="generalization_test1.txt"):
    dataA = open(fileA,"r")
    dataB = open(fileB,"r")
    linesA = dataA.readlines()
    linesB = dataB.readlines()

    #正解数
    count = 0
    test_num = 10000
    for i in range(test_num):
        if linesA[i] == linesB[i]:
            count += 1

    print(count/test_num)

if __name__ == '__main__':
    compare()
