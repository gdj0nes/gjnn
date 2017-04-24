from __future__ import print_function

import time

import numpy as np


class Model(object):
    """Container for a feed forward neural network
    """

    def __init__(self, optimizer, epoch=0):
        """
        Parameters
        ----------
        optimizer: An object used for updating the learnable weights of the matrix

        """
        self.epoch = epoch
        self.optimizer = optimizer
        self.layers = []  # Container of layers
        self.history = {"train_loss": [],
                        "val_loss": [],
                        'train_acc': [],
                        'val_acc': [],
                        'time': []}

    def add(self, layer):
        """Used to add layers to the model
        """
        self.layers.append(layer)

    def train(self, train_data, val_data, batch_size, num_epochs, verbose=True):
        """
        The training method for updating the model. The data for training is stored in an h5 dataset to
        reduce the amount of memory utilized. The H5 must have the following datasets: trainX, trainY, 
        testX, testY. 

        Parameters:
        ----------
        path (str): file path to h5 dataset
        batch_size (int): the number of examples used in a mini-batch
        num_epochs (int): the number of training epochs

        """

        last_epoch = self.epoch + num_epochs  # Update epoch counter

        X_train, Y_train = train_data
        X_val, Y_val = val_data
        num_obs = X_train.shape[0]
        num_val = X_val.shape[0]

        # Early stopping
        top_val = 0
        val_counter = 0

        while self.epoch < last_epoch:
            t0 = time.time()
            if verbose: print('Epoch:', self.epoch, end="\n   ")
            self.epoch += 1
            # Train Stage
            train_loss = 0.
            train_acc = 0.
            evals = 0
            for batch_num in xrange(num_obs / batch_size):
                # Construct batch
                start = batch_size * batch_num
                batch = X_train[start:start + batch_size]
                batch_labels = Y_train[start:start + batch_size]
                # Forward and backward
                self.forward(batch, train=True)
                self.backward(batch_labels)
                # Collect loss and accuracy
                if start % batch_size * 100 == 0:
                    evals += 1
                    batch_results = self.evaluate_training(batch_labels)
                    train_loss += batch_results[0]
                    train_acc += batch_results[1]
                    if verbose:
                        print("Progress: {:.0%} Loss: {:.4f} Accuracy: {:.4f}".format(start / float(num_obs),
                                                                                      batch_results[0],
                                                                                      batch_results[1]), end='\r')

            # Report and save epoch train results
            batch_num += 1  # Adjust for taking averagge
            train_loss /= evals
            train_acc /= evals
            # Epoch training time
            t1 = time.time()
            total_n = t1 - t0
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history['time'].append(total_n)
            if verbose:
                print()
                print("Training:  Loss: {:.4f} Accuracy: {:.4f} Time: {:.2f}s".format(train_loss,
                                                                                      train_acc,
                                                                                      total_n))
                # Validation stage
            val_loss = 0.
            val_acc = 0.
            for batch_num in xrange(num_val / batch_size):
                start = batch_size * batch_num
                batch = X_val[start:start + batch_size]
                batch_labels = Y_val[start:start + batch_size]
                batch_results = self.evaluate_valid(batch, batch_labels)
                val_loss += batch_results[0]
                val_acc += batch_results[1]
            # Report and save epoch validation results
            batch_num += 1
            val_loss /= batch_num
            val_acc /= batch_num
            if verbose:
                print("Validation: Loss: {:.4f}  Accuracy: {:.4f}".format(val_loss, val_acc))
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            # Early stopping conditions
            if val_acc - 0.001 < top_val:
                val_counter += 1
                if val_counter == 5:
                    break
            else:
                top_val = val_acc
                val_counter = 0

        self.inputs = None  # clear model of values
        return self.history

    def forward(self, batch, train=False):
        """Preform a forward pass over the layers in the model

        Parameters:
            :param batch: the data for batch 
            :param train: optional indicator for dropout layers

        """
        self.inputs = [batch]
        for layer in self.layers:
            layer.train = train
            prev_output = self.inputs[-1]
            layer_output = layer.forward(prev_output)
            self.inputs.append(layer_output)

        return self.inputs[-1]

    def backward(self, labels):
        """Preform backpropgation and parameter updating
        """

        INPUT_COR = 1 + 1  # Adujustment for fetching inputs
        output = self.inputs[-1]  # Get output
        loss_layer = self.layers[::-1][0]  # Compute data loss
        output_grad = loss_layer.backward(output, labels)  # Compute gradient on loss
        for i, layer in enumerate(self.layers[::-1][1:]):
            layer_input = self.inputs[::-1][i + INPUT_COR]  # Get the input passed the layer
            output_grad = layer.backward(layer_input, output_grad)
            layer.update(self.optimizer)  # Update layer weights

    def evaluate_training(self, labels):
        """Evaluate training loss and accuracy for most recent batch. Utilizes output from previous
        pass tp 
        """

        output = self.inputs[-1]
        loss = self.layers[-1].evaluate(output, labels)
        preds = output.argmax(axis=1)
        acc = (labels == preds).sum() / float(len(labels))
        return loss, acc

    def evaluate_valid(self, data, labels):
        """Evaluate validation loss and accuracy for most recent batch. Computes predictions
        """

        output = self.forward(data)
        loss = self.layers[-1].evaluate(output, labels)
        preds = output.argmax(axis=1)
        acc = (labels == preds).sum() / float(len(labels))
        return loss, acc

    def saveModel(self):
        """Future work
        """

        pass

    def loadModel(self):
        """Future work
        """
        pass

    def predict(self, data):
        """
        TODO: Make this able to be done in batch if number of examples excedes threhsold
        """
        rv = np.argmax(self.forward(data), axis=1)
        return rv
