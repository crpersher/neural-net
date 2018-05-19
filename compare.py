import sys, os
sys.path.append(os.pardir)
import numpy as np

def compare(fileA = "generalization_test0.txt", fileB ="generalization_test1.txt",regression = False):
    dataA = open(fileA,"r")
    dataB = open(fileB,"r")
    linesA = dataA.readlines()
    linesB = dataB.readlines()
    test_num = 10000
    count = 0
    if(regression):
        for i in range(len(linesA)):
                count = count + abs(float(linesA[i]) - float(linesB[i]))
    else:
    #正解数
        for i in range(test_num):
            if linesA[i] == linesB[i]:
                count += 1

    print(count/test_num)

if __name__ == '__main__':
    compare()
