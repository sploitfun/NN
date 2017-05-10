from sklearn import datasets
import numpy as np

import os
import gzip
import cPickle

def conv_y(y): 
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e

def load_mnist_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    train_targets = [conv_y(y) for y in train_set[1]]
    train_data = zip(train_inputs, train_targets)
    
    valid_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    valid_targets = valid_set[1]
    valid_data = zip(valid_inputs, valid_targets)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    test_targets = test_set[1]
    test_data = zip(test_inputs, test_targets)
    return train_data, valid_data, test_data

def predict(params, x):
    W1 = params[0]
    b1 = params[1]
    W2 = params[2]
    b2 = params[3]
    #Forward Propogation
    a0 = x
    z1 = W1.dot(a0) + b1
    a1 = sigmoid(z1)
    z2 = W2.dot(a1) + b2
    a2 = sigmoid(z2)
    #Returns Output Unit Index
    return np.argmax(a2)

def validate(params, valid_data):
    valid_results = [(predict(params, x) == y) for x, y in valid_data]
    return sum(valid_results)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z):
    return sigmoid(z) * (1 - sigmoid(z))

def build_nn(train_data, valid_data):
    #NN Arch Init
    inp_units = len(train_data[0][0])
    hid_units = 25
    out_units = len(train_data[0][1])
    
    #Random Weight Init
    W1 = np.random.randn(hid_units, inp_units)
    W2 = np.random.randn(out_units, hid_units)
    b0 = np.random.randn(inp_units, 1)
    b1 = np.random.randn(hid_units, 1)
    b2 = np.random.randn(out_units, 1)

    #HyperParameters Init
    epsilon = 1.0
    mini_batch_sz = 32
    iterations = 10

    for iteration in range(0, iterations):
        mini_batches = [train_data[i:i + mini_batch_sz] for i in range(0, len(train_data), mini_batch_sz)]
        for mini_batch in mini_batches:
            DW2 = np.zeros((out_units, hid_units))
            DW1 = np.zeros((hid_units, inp_units))
            Db2 = np.zeros((out_units, 1))
            Db1 = np.zeros((hid_units, 1))
            for x, y in mini_batch:
                #Forward Propogation
                a0 = x
                z1 = W1.dot(a0) + b1
                a1 = sigmoid(z1)
                z2 = W2.dot(a1) + b2
                a2 = sigmoid(z2)
                #Backward Propogation
                #delta2 = np.multiply(-(y-a2), sigmoid_grad(z2))
                #delta1 = np.dot(W2.T,delta2)*sigmoid_grad(z1)
                delta2 = (a2 - y)
                delta1 = np.multiply(W2.T.dot(delta2), sigmoid_grad(z1))
                dW2 = delta2.dot(a1.T) 
                dW1 = delta1.dot(a0.T)
                db2 = delta2
                db1 = delta1
                DW2 = DW2 + dW2
                DW1 = DW1 + dW1
                Db2 = Db2 + db2
                Db1 = Db1 + db1
            #Simultaneous Parameter Update
            W1 = W1 - (epsilon / mini_batch_sz) * DW1
            W2 = W2 - (epsilon / mini_batch_sz) * DW2
            b1 = b1 - (epsilon / mini_batch_sz) * Db1
            b2 = b2 - (epsilon / mini_batch_sz) * Db2

        params = (W1, b1, W2, b2)

        if valid_data:
            pred_accuracy = validate(params, valid_data) / 100.0
            print "NN Accuray (on Validation Data) after iteration %d: %f %%" %(iteration+1, pred_accuracy)

    return params

train_data, valid_data, test_data = load_mnist_data()

print("Building a single hidden layer (with 25 hidden units) neural network model...")
params = build_nn(train_data, valid_data)

#Testing NN Model
print "Target Num:%d, Output Num:%d" %(test_data[0][1], predict(params, test_data[0][0]))
print "Target Num:%d, Output Num:%d" %(test_data[100][1], predict(params, test_data[100][0]))
print "Target Num:%d, Output Num:%d" %(test_data[500][1], predict(params, test_data[500][0]))
print "Target Num:%d, Output Num:%d" %(test_data[1000][1], predict(params, test_data[1000][0]))
print "Target Num:%d, Output Num:%d" %(test_data[5000][1], predict(params, test_data[5000][0]))
