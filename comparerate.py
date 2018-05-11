import sys, os
sys.path.append(os.pardir)
import numpy as np
import compare
import train_neuralnet

train_num1 = 1000
epoch_num1 = 500
hidden_num1 = 30
batch_size1 = 100
learning_rate1 = 0.07

train_num2 = 1000
epoch_num2 = 501
hidden_num2 = 30
batch_size2 = 100
learning_rate2 = 0.07

for i in range(10):
    train_neuralnet.train_neuralnet(train_num1,epoch_num1,hidden_num1,batch_size1,learning_rate1)
    train_neuralnet.train_neuralnet(train_num2,epoch_num2,hidden_num2,batch_size2,learning_rate2)

    compare.compare("train_" + str(train_num1) + "_epoch_" + str(epoch_num1) + "_hidd_" + str(hidden_num1) + "_batch_" + str(batch_size1) + "_learn_" + str(learning_rate1) + ".txt"\
    ,"train_"+ str(train_num2) +"_epoch_"+ str(epoch_num2) + "_hidd_" + str(hidden_num2) + "_batch_" + str(batch_size2) + "_learn_" + str(learning_rate2) + ".txt")
