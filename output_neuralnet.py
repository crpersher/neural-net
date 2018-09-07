# coding: utf-8
import sys, os
import pickle
import random

sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet



def output_neuralnet(train_num = 60000,epoch_num = 1000,hidden_num = 300,batch_size = 100,learning_rate = 0.1,data_num = 1,ratechange = False,samecompare_epoch = 0,comparevalues = False,seed = 0):

    see_weight = False
    see_acc   = True
    data2 = 2

    random_change = True
    random.seed(seed)

    # データの読み込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape)
    x_train = x_train[:train_num]
    t_train = t_train[:train_num]
    train_size = x_train.shape[0]
    #学習データのランダム化
    if(random_change):
        for i in range(train_size):
            a = random.randrange(10)
            for j in range(10):
                if(j == a):
                    t_train[i][j] = 1
                else:
                    t_train[i][j] = 0

    #重みを表示する
    if(see_weight):
        w11 = []
        w12 = []
        w21 = []
        w22 = []


    network = TwoLayerNet(input_size=784, hidden_size=hidden_num, output_size=10)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

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
        if(see_weight):
            if(i < iters_num/2):
                w11.append(network.params['W1'][300][5])
                w12.append(network.params['W1'][300][6])
            else:
                w21.append(network.params['W1'][300][5])
                w22.append(network.params['W1'][300][6])

        #print(w1)
        #print(w2)
        if i % (iter_per_epoch*100) == 0:
            #sameに値が入っている場合は二つファイルを作成する必要がある．
            if(samecompare_epoch > 0 and i / iter_per_epoch == samecompare_epoch):
                    y = network.predict(x_test)
                    y = np.argmax(y, axis=1)
                    f = open('data' + str(data2) + '.txt','w')
                    count = 0
                    for i in y:
                        f.write(str(i) + "\n")
                        count += 1
                    f.write("count:" + str(count))

            if(see_acc):
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                #print(str(int(i/iter_per_epoch)) + ":" + str(train_acc) + str(test_acc))
                print('{0} : {1:.4f}  {2:.4f}'.format(int(i/iter_per_epoch),train_acc,test_acc))

    y = network.predict(x_test)
    #クラス認識をする場合はコメント外す
    if(comparevalues):
        y = y.reshape(-1,)
    else:
        y = np.argmax(y, axis=1)
    f = open('data' + str(data_num) +'.txt','w')
    count = 0
    for i in y:
        f.write(str(i) + "\n")
        count += 1

    if(see_weight):
        plt.plot(w11,w12,"ro")
        plt.plot(w21,w22,"o")
        plt.show()


if __name__ == '__main__':
    train_neuralnet()
