# coding: utf-8
import sys, os
import pickle

sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

def regression(train_num = 1000,epoch_num = 600,hidden_num = 30,batch_size = 1,learning_rate = 0.1,ratechange = False):

    seeweight = False
    see_acc = True
    network = TwoLayerNet(input_size=1, hidden_size=hidden_num, output_size=1)

    x_train = np.array([-3,-2,-1,0,1,2,3])
    t_train = np.array([[1],[1],[0],[1],[0],[0],[1]])
    x_test = np.linspace(-5,5,10)
    train_size = x_train.shape[0]
    train_loss_list = []
    train_acc_list = []

    batch_size = train_size
    iter_per_epoch = max(train_size / batch_size, 1)
    iters_num = int(iter_per_epoch * epoch_num)

    for i in range(iters_num):

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 勾配
        #grad = network.numerical_gradient(x_batch, t_batch)
        grad = network.gradient(x_batch, t_batch)

        # 更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        #途中で学習率の変更
        if(ratechange and (i > iters_num/2)):
            learning_rate = learning_rate / 2
            ratechange = False
        #重みの表示用登録
        if(seeweight):
            if(i < iters_num/2):
                w11.append(network.params['W1'][0][5])
                w12.append(network.params['W1'][0][6])
            else:
                w21.append(network.params['W1'][0][5])
                w22.append(network.params['W1'][0][6])

        #print(w1)
        #print(w2)
        if i % (iter_per_epoch*100) == 0:
            #sameに値が入っている場合は二つファイルを作成する必要がある．

            if(see_acc):
                train_acc = network.accuracy(x_train, t_train)
                train_acc_list.append(train_acc)
                #print(str(int(i/iter_per_epoch)) + ":" + str(train_acc) + str(test_acc))
                print('{0} : {1:.4f} '.format(int(i/iter_per_epoch),train_acc))

    #重みの表示
    if(seeweight):
        plt.plot(w11,w12,"ro")
        plt.plot(w21,w22,"o")
        plt.show()
    #テストデータの可視化
    y = network.predict(x_test)
    print(y)
    plt.plot(x_test,y)
    plt.show()

if __name__ == '__main__':
    regression()
