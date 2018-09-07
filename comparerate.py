import sys, os
sys.path.append(os.pardir)
import numpy as np
import compare
import output_neuralnet

train_num = 60
epoch_num = 100
hidden_num = 10
batch_size = 100
learning_rate = 0.1

ratechange = False
comparevalues = False


for i in range(5):

    output_neuralnet.output_neuralnet(train_num,epoch_num,hidden_num,batch_size,learning_rate,data_num = 1,ratechange = ratechange,comparevalues = comparevalues,seed = 5)
    output_neuralnet.output_neuralnet(train_num,epoch_num,hidden_num,batch_size,learning_rate,data_num = 2,ratechange = ratechange,comparevalues = comparevalues,seed = 5)
    compare.compare(fileA="data1.txt",fileB="data2.txt",comparevalues = comparevalues)
