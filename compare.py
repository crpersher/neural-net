import sys, os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np

dataA = open("generalization_test0.txt","r")
dataB = open("generalization_test1.txt","r")
linesA = dataA.readlines()
linesB = dataB.readlines()

#正解数
count = 0
for i in range(10000):
    if linesA[i] == linesB[i]:
        count += 1

print(count)
