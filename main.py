import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle as pkl

import os

from model import CNN2019
from data_loader import load_data
from cnn_train import CNNTrainer
# from evaluate import Evaluator

def make_model(data, opts):
	channels, num_classes, train_loader, val_loader, test_loader = load_data(train_data, val_data, test_data, opts)
	if opts.model =='cnn2019':
		model = CNN2019(in_channels, num_classes, avg_pool=opts.avg_pool, conv_dim=opts.conv_dim, hs=opts.hs)
	else:
		raise NotImplementedError
	if opts.cuda:
		model.cuda()
		print('Using GPU')

	return model, train_loader, test_loader

def cnn_train(model, train_loader, test_loader, opts):
	optimizer = optim.Adam(model.parameters(), lr=opts.lr, amsgrad=True)
	loss = nn.CrossEntropyLoss()

	if opts.load_from_chkpt != None:
		checkpoint = torch.load('/'.join([opts.base_path,'runs',opts.run,'checkpoints',opts.load_from_chkpt]))
		model.load_state_dict(checkpoint['state_dict'])
		opts.start_epoch = checkpoint['epoch']
		print("Start epoch", opts.start_epoch)
		optimizer.load_state_dict(checkpoint['optimizer'])

	trainer = CNNTrainer(model, optimizer, loss, train_loader, test_loader, opts)

	trainer.train(opts)

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', dest='mode', default='train', help='Choose from train, eval')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--base_path', default='.')
	parser.add_argument('--run', default='run')

	# model params
	parser.add_argument('--model', default='cnn2019', help='Choose from cnn2019')
	parser.add_argument('--conv_dim', type=int, default=32, 'Num filters for the cnn2019 -- each layer has same # filters')
	parser.add_argument('--hs', default=[64,32], nargs='*', type=int, help='List of')
	parser.add_argument('--avg_pool', default=1, nargs='*', type=int, help='What to global average pool to per channel, default 1')

	# training params
	parser.add_argument('--epochs', dest='epochs', type=int, default = 10000)
	parser.add_argument('--test_every', dest='test_every', type = int, default = 2)
	parser.add_argument('--checkpoint_every', dest='checkpoint_every', type = int, default = 4)
	parser.add_argument('--load_from_chkpt', dest='load_from_chkpt', default=None) # Also used for eval
	parser.add_argument('--new_chkpt_fname', dest='new_chkpt_fname', default="checkpoint.pth.tar")
	parser.add_argument('--lr', dest='lr', type=float, default=0.001)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--start_epoch', type=int, default=0)
	# parser.add_argument('--lr_decay', dest='lr_decay', type=float, default=0.99)
	# parser.add_argument('--lr_step', dest='lr_step', type=int, default=1)
	# parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.3)

	# data params
	parser.add_argument('--num_workers', type=int, default=1)
	parser.add_argument('--data_files', nargs='*', default=['dataset.pkl'])
	parser.add_argument('--balance_classes', type=int, default=1)

	# audio process params
	parser.add_argument('--transform', default='mfcc', help='transform to perform on audio, choose from mfcc,stft,raw')
	parser.add_argument('--window_size', type=int, default=25, help='window size (in ms) for above transform')
	parser.add_argument('--slide', type=int, default=10, help='Amount to slide window (in ms) for above transform')
	parser.add_argument('--numceps', type=int, default=40, help='Number of cepstral filters/features to keep')


	# Evaluation arguments
	# parser.add_argument('--eval_type', dest='eval_type', default='all', help='Choose from randgen, cluster, transform, real_cluster, all')
	# # Note for eval, we always use the Res18R1 CNN trained on that dataset
	# parser.add_argument('--op_samples', dest='op_samples', type=int, default=400, help='Number of images to do eval_type on and plot')
	# parser.add_argument('--pca_components', dest='pca_components', type=int, default=6, help='Number of PCA components to show')
	# parser.add_argument('--random_pick', dest='random_pick', type=int, default=0, help="Whether or not to pick images to interpolate randomly")
	# # TODO make intermediate a passable parameter... not enough time rn tho
	# parser.add_argument('--intermediate', dest='intermediate', type=int, default=16, help='How many intermediate images to have in interpolation')

	return parser


if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()
	print(opts.data_path)
	# print("Using movements", opts.movements)
	opts.cuda = 1 if torch.cuda.is_available() else 0
	# counts = [0, 0]
	# for i in range(2):
	# 	for batch_idx, (data, labels) in enumerate(train_loader):
	# 		print(labels)
	# 		if batch_idx > 5:
	# 			break
	# 	print('bla')
	# print(batch_idx)
	# print(counts)
	with open(data_file, 'rb') as f:
		data = pkl.load(f)
	if opts.mode == 'train':
		model, train_loader, val_loader, _ = make_model(data, opts)
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
