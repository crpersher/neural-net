import sys, os
sys.path.append(os.pardir)
import numpy as np
import compare
import train_neuralnet

train_num = 500
epoch_num = 600
hidden_num = 200
batch_size = 100
learning_rate = 0.1

ratechange = False
regression = False


for i in range(5):
    train_neuralnet.train_neuralnet(train_num,epoch_num,hidden_num,batch_size,learning_rate,data_num = 1,ratechange = ratechange,regression = regression,seed = 5)
    train_neuralnet.train_neuralnet(train_num,epoch_num,hidden_num,batch_size,learning_rate,data_num = 2,ratechange = ratechange,regression = regression,seed = 5)
    compare.compare(fileA="data1.txt",fileB="data2.txt",regression = regression)
