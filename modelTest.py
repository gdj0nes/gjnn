import h5py
from activations import softmax, tanh
from layers import FullyConnected, DropOut
from models import Model
from optimizers import SGD, RMSProp
from weights import xavier_init

with h5py.File('mnist.h5', 'r') as h5:
    X_train = h5['trainX'].value
    y_train = h5['trainY'].value
    X_val = h5['testX'].value
    y_val = h5['testY'].value

hidden1_size = 300
hidden2_size = 300
learning_rate = 0.5
decay = 0.0005


model = Model(SGD(learning_rate))
for layer in layers:
    layers = [
        FullyConnected(784, hidden1_size, xavier_init),
        tanh(),
        DropOut(0.9),
        FullyConnected(hidden1_size, hidden2_size, xavier_init),
        tanh(),
        DropOut(0.25),
        FullyConnected(hidden2_size, 10, xavier_init),
        softmax()
    ]
    model.add(layer)

_ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=10, verbose=1)
# #
# model = Model(SGD(learning_rate))
# layers = [
#     FullyConnected(784, hidden1_size, xavier_init, l2=decay),
#     tanh(),
#     FullyConnected(hidden1_size, hidden2_size, xavier_init, l2=decay),
#     tanh(),
#     FullyConnected(hidden2_size, 10, xavier_init, l2=decay),
#     softmax()
# ]
# for layer in layers:
#     model.add(layer)
#
# _ = model.train((X_train, y_train), (X_val, y_val), batch_size=256, num_epochs=1, verbose=1)

# decay = 0.0005
# model = Model(RMSProp(learning_rate))
# layers = [
#     FullyConnected(784, hidden1_size, xavier_init),
#     tanh(),
#     FullyConnected(hidden1_size, hidden2_size, xavier_init),
#     tanh(),
#     FullyConnected(hidden2_size, 10, xavier_init),
#     softmax()
# ]
# for layer in layers:
#     model.add(layer)
#
# _ = model.train((X_train, y_train), (X_val, y_val), batch_size=55, num_epochs=10, verbose=1)
