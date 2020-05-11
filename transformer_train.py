from torch.autograd import Variable

import shutil
from tqdm import tqdm
import numpy as np

import os
import pdb
import pickle
import argparse

import warnings
warnings.filterwarnings("ignore")

import math

# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# Numpy & Scipy imports
import numpy as np
import scipy
import scipy.misc
from data_loader import load_data

device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelTrainer:
    def __init__(self, model, optimizer, loss, train_loader, test_loader, opts):
        self.model = model
        self.opts = opts
        self.cuda = opts.cuda

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss = loss
        self.optimizer = optimizer

        print("Using cuda?", next(model.parameters()).is_cuda)

    def train(self, opts):
        self.model.train()
        best_loss = 1e12
        best_accuracy = 0
        for epoch in range(self.opts.start_epoch, self.opts.epochs):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader)):
                if self.cuda:
                    #                    print('using GPU')
                    data = data.cuda()
                    labels = labels.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                linear_pred = self.model(data)
                loss = self.loss(linear_pred, labels)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            epoch_avg_loss = np.mean(loss_list) / self.opts.batch_size
            print("epoch {}: - training loss: {}".format(epoch, epoch_avg_loss))

            if epoch % opts.test_every == 0:
                new_loss, new_accuracy = self.test(epoch)
                print("epoch {}: - test loss: {} - test acc: {}".format(epoch, new_loss, new_accuracy))
                if (new_loss < best_loss) or (new_accuracy > best_accuracy):
                    print("found new best model on epoch {}".format(epoch))
                    best_loss = new_loss
                    best_accuracy = new_accuracy

    def test(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        correctly_classified = 0
        for i, (data, labels) in enumerate(self.test_loader):
            if self.cuda:
                data = data.cuda()
                labels = labels.cuda()
            data = Variable(data)
            linear_pred = self.model(data)
            test_loss += self.loss(linear_pred, labels).item()
            correctly_classified += (linear_pred.max(dim=1)[1] == labels).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = correctly_classified / len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set accuracy: {:.6f}'.format(accuracy))
        self.model.train()
        return test_loss, accuracy


class LSTM_Classifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0., bidirectional=False):
    super(LSTM_Classifier, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.bidirectional = bidirectional
    self.dropout = dropout

    self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                      dropout=dropout, bidirectional=bidirectional)
    self.linear = nn.Linear(hidden_size + hidden_size * int(bidirectional), output_size)
    self.out = nn.LogSoftmax(dim=1)

  def forward(self, input):
    rnn_out, _ = self.rnn(input)
    linear_out = self.linear(rnn_out)
    return self.out(linear_out)


if __name__ == "__main__":
	all_data = dict()
	for data_file in opts.data_files:
		with open(data_file, 'rb') as f:
			data = pkl.load(f)
		for k in data:
			all_data.setdefault(k,[]).extend(data[k])

	np.random.seed(0) # TODO consistent random seeding for torch
	train_data = dict()
	val_data = dict()
	test_data = dict()
	for k in all_data:
		if len(all_data[k]) > opts.min_class_size:
			np.random.shuffle(all_data[k])
			num_train = int(opts.train_split*(len(all_data[k])-opts.num_test_samples))
			train_data[k] = all_data[k][opts.num_test_samples:num_train+opts.num_test_samples]
			val_data[k] = all_data[k][num_train+opts.num_test_samples:]
			test_data[k] = all_data[k][:opts.num_test_samples]
	print("num classes", len(train_data))
	if opts.mode == 'train':
		model, train_loader, val_loader, _ = make_model(train_data,val_data,test_data, opts)
		cnn_train(model, train_loader, val_loader, opts)
	elif opts.mode == 'eval':
		raise NotImplementedError
		# opts.cuda = 0
		# opts.batch_size = opts.op_samples
		# model, _, _, test_loader = make_model(opts)
		# classifier, _, _ = make_cnn(opts)
		# eval(model, classifier, test_loader, opts)
	else:
		raise NotImplementedError
	# gen_image('./runs/checkpoints/checkpoint2.pth.tar')
