import sys, os
sys.path.append(os.pardir)
import numpy as np
import compare
import train_neuralnet

train_num1 = 1000
epoch_num1 = 600
hidden_num1 = 30
batch_size1 = 100
learning_rate1 = 0.1

ratechange = False

for i in range(10):
    train_neuralnet.train_neuralnet(train_num1,epoch_num1,hidden_num1,batch_size1,learning_rate1,ratechange = ratechange,samecompare_epoch = 500)

    compare.compare("data1.txt","data2.txt")
