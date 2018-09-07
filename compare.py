
import sys, os
sys.path.append(os.pardir)
import numpy as np

def compare(fileA = "data1.txt", fileB ="data2.txt",comparevalues = False):
    dataA = open(fileA,"r")
    dataB = open(fileB,"r")
    linesA = dataA.readlines()
    linesB = dataB.readlines()
    if(len(linesA) != len(linesB)):
        print("2 files are different length.")
    test_num = len(linesA)
    count = 0
    if(comparevalues):
        for i in range(test_num):
                count = count + abs(float(linesA[i]) - float(linesB[i]))
    else:
    #rate of concordance
        for i in range(test_num):
            if linesA[i] == linesB[i]:
                count += 1

    print(count/test_num)

if __name__ == '__main__':
    compare()
