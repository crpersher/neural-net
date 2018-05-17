import sys, os
sys.path.append(os.pardir)
import numpy as np
import compare
import train_neuralnet

train_num1 = 1000
epoch_num1 = 1500
hidden_num1 = 30
batch_size1 = 1000
learning_rate1 = 0.5

train_num2 = 1000
epoch_num2 = 1500
hidden_num2 =30
batch_size2 = 1000
learning_rate2 = 0.5

ratechange = False

for i in range(10):
    train_neuralnet.train_neuralnet(train_num1,epoch_num1,hidden_num1,batch_size1,learning_rate1,ratechange = ratechange)
    train_neuralnet.train_neuralnet(train_num2,epoch_num2,hidden_num2,batch_size2,learning_rate2,data1 = 2,ratechange = ratechange)

    compare.compare("data1.txt","data2.txt")
