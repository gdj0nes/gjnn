import h5py
from activations import Softmax, Tanh, ReLU
from layers import FullyConnected, DropOut
from models import Model
from optimizers import SGD, RMSProp, Adam
from weights import xavier_init

with h5py.File('mnist.h5', 'r') as h5:
    X_train = h5['trainX'].value
    y_train = h5['trainY'].value
    X_val = h5['testX'].value
    y_val = h5['testY'].value

hidden1_size = 100
hidden2_size = 100
learning_rate = 0.005
decay = 0.0005

dropout = False
decay = True
rmsprop = False
adam = False

# TODO: Test values with tensorflow

#### TEST DROPOUT  ####
if dropout:
    model = Model(SGD(learning_rate))
    for layer in layers:
        layers = [
            FullyConnected(784, hidden1_size, xavier_init),
            Tanh(),
            DropOut(0.9),
            FullyConnected(hidden1_size, hidden2_size, xavier_init),
            Tanh(),
            DropOut(0.25),
            FullyConnected(hidden2_size, 10, xavier_init),
            Softmax()
        ]
        model.add(layer)

    _ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=2, verbose=1)

#### TEST DECAY ####
if decay:
    model = Model(SGD(0.05))
    layers = [
        FullyConnected(784, hidden1_size, xavier_init, l2=decay),
        Tanh(),
        FullyConnected(hidden1_size, hidden2_size, xavier_init, l2=decay),
        Tanh(),
        FullyConnected(hidden2_size, 10, xavier_init, l2=decay),
        Softmax()
    ]
    for layer in layers:
        model.add(layer)

    _ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=5, verbose=1)

#### TEST RMSPROP ####
if rmsprop:
    model = Model(RMSProp(learning_rate))
    layers = [
        FullyConnected(784, hidden1_size, xavier_init),
        ReLU(),
        DropOut(0.25),
        FullyConnected(hidden1_size, hidden2_size, xavier_init),
        ReLU(),
        DropOut(0.25),
        FullyConnected(hidden2_size, 10, xavier_init),
        Softmax()
    ]
    for layer in layers:
        model.add(layer)

    _ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=5, verbose=1)

#### TEST ADAM ####
if adam:
    model = Model(Adam(learning_rate))
    layers = [
        FullyConnected(784, hidden1_size, xavier_init),
        ReLU(),
        FullyConnected(hidden1_size, hidden2_size, xavier_init),
        ReLU(),
        FullyConnected(hidden2_size, 10, xavier_init),
        Softmax()
    ]
    for layer in layers:
        model.add(layer)

    _ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=5, verbose=1)
