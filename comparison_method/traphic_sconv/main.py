import argparse
import warnings
import re
import os
import csv
from model.model import TnpModel
from model.import_data import *
import time

warnings.filterwarnings("ignore")


# formatted_hypo_file (5 cols):
# dset_idx, vid_idx, frames_idx, tl-X, tl-Y
#
# npy file for training:
# -- traj: 0 -47
# 0-4: same as hypo_file
# see import_data.py for 5 - 47 (features)
#
# -- track:
# vid -> [frame_number, tl-X, tl-Y]
#
#
# txt file for training (5 cols):
# frames_idx, vid_idx, tl-X, tl-Y, dset_idx


# dataset dirs
DATASET = 'ARGO'
# DATASET = 'LYFT'
# DATASET = 'APOL'
LOG = './logs/'
# LOAD = 'Traphic_LYFT_model_30-50l_12em.tar'
LOAD = ''
CUDA = True 
# CUDA = False
DEVICE = 'cuda:0'
# DEVICE = 'cuda:1'
PREDALGO = 'Traphic'
# PREDALGO = 'Social_Conv'
PRETRAINEPOCHS= 6
TRAINEPOCHS= 20
INPUT = 20
OUTPUT = 30
NAME = '{}_{}' + '_model_{}-{}l_{}e.tar'.format(INPUT, OUTPUT, PRETRAINEPOCHS + TRAINEPOCHS)
# NAME = 'Traphic_LYFT_model_30-50l_12em.tar'
TENSORBOARD = False

DATA_DIR = '../../resources/data/' + DATASET

MODELLOC = "../../resources/trained_models"
RAW_DATA = "../../resources/raw_data/" + DATASET

TRAIN = False
EVAL = True
# training option
BATCH_SIZE = 10
DROPOUT = 0.5
OPTIM= 'Adam'
# SGD Adam AdamW SparseAdam Adamax ASGD RMSprop Rprop 
LEARNING_RATE= 0.01
MANEUVERS = False
PRETRAIN_LOSS = 'NLL'
TRAIN_LOSS = 'MSE'

# DSET_IDX = 0
# VID_IDX = 1
# FRAMES_IDX = 2
# X = 3
# Y = 4

def argo_to_formatted(input_dir, files, output_dir, dtype):
	txtlst = []
	i = 0 
	sz = len(files)
	for f in files:
		print("Processing {}/{} in {}...".format(i, sz, dtype))
		i += 1
		dset_id = f.split('.')[0]

		out_name = dset_id + '.txt'
		txtlst.append(dset_id)

		current_time = -1
		current_frame_num = -1

		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		out = open(os.path.join(output_dir, out_name), 'w')
		f = os.path.join(input_dir, f)

		with open(f) as csv_file:
			for row in csv.reader(csv_file):
				if row[0] == 'TIMESTAMP':
					continue;
				if float(row[0]) > current_time:
					current_time = float(row[0])
					current_frame_num += 1
				vid = int(row[1].split('-')[-1])
				out.write("{},{},{},{},{}\n".format(dset_id, vid, current_frame_num, row[3], row[4]))


	return txtlst

def create_data(input_dir, file_names, output_dir, dtype, threadid):
	
	name_lst = []
	i = 0
	sz = len(file_names)
	for f in file_names:
		print("Importing data {}/{} for {} in thread {}...".format(i, sz, dtype, threadid))
		i += 1
		loc = os.path.join(input_dir,f+'.txt')
		out = os.path.join(input_dir,f+'.npy')
		import_data(loc, None, out)
		name_lst.append(out)
	# merge_n_split(name_lst, output_dir)
	merge(name_lst, output_dir, dtype, threadid)



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="traphicPred command line control")

	parser.add_argument('--cuda', '-g', action='store_true', help='GPU option', default=CUDA)
	parser.add_argument('--device', '-d', help='cuda device option', default=DEVICE, type=str)

	parser.add_argument('--batch_size', '-b', help='bastch size', default=BATCH_SIZE)
	parser.add_argument('--dropout', help='dropout probability', default=DROPOUT)
	parser.add_argument('--lr', help='learning rate', default=LEARNING_RATE)
	parser.add_argument('--optim', help='optimiser', default=OPTIM)
	parser.add_argument('--pretrainEpochs', '-p', help='number of epochs for pretraining', default=PRETRAINEPOCHS, type=int)
	parser.add_argument('--trainEpochs', '-e', help='number of epochs for training', default=TRAINEPOCHS, type=int)
	parser.add_argument('--maneuvers', help='maneuvers option', default=MANEUVERS, type=bool)
	parser.add_argument('--predalgo', help='prediction algorithm', default=PREDALGO)
	parser.add_argument('--pretrain_loss', help='pretrain loss algorithm', default=PRETRAIN_LOSS)
	parser.add_argument('--train_loss', help='train loss algorithm', default=TRAIN_LOSS)

	parser.add_argument('--dset', '-s', help='cuda device option', default=DATASET, type=str)
	parser.add_argument('--modelLoc', help='trained prediction store/load location', default=MODELLOC)
	parser.add_argument('--dir', help="location of the dataset for tracking", default=DATA_DIR)



	args = parser.parse_args()



	viewArgs = {}
	viewArgs['cuda'] = args.cuda
	viewArgs['log_dir'] = LOG
	viewArgs['batch_size'] = args.batch_size
	viewArgs['dropout'] = args.dropout
	viewArgs["lr"] = args.lr
	viewArgs["optim"] = args.optim
	viewArgs['pretrainEpochs'] = args.pretrainEpochs
	viewArgs['trainEpochs'] = args.trainEpochs
	viewArgs["maneuvers"] = args.maneuvers
	viewArgs['predAlgo'] = args.predalgo
	viewArgs['pretrain_loss'] = args.pretrain_loss
	viewArgs['train_loss'] = args.train_loss

	viewArgs['tensorboard'] = TENSORBOARD

	viewArgs['modelLoc'] = args.modelLoc
	viewArgs['dir'] = args.dir
	viewArgs['raw_dir'] = RAW_DATA
	if not args.cuda:
		args.device = 'cpu'
	viewArgs['device'] = args.device

	viewArgs['dsId'] = 0
	viewArgs['dset'] = args.dset
	viewArgs['name_temp'] = NAME
	viewArgs['input_size'] = INPUT
	viewArgs['output_size'] = OUTPUT
	# lst = ['formatted_hypo']
	# create_data(RAW_DATA, lst, args.dir)

	# train_loc = RAW_DATA + '/train/data/'
	# output_dir = RAW_DATA + '/train/formatted/'
	# files = [f for f in os.listdir(train_loc) if '.csv' in f]
	# train_lst = argo_to_formatted(train_loc, output_dir, "train")
	# create_data(output_dir, train_lst, args.dir, "train")


	# val_loc = RAW_DATA + '/val/data/'
	# output_dir = RAW_DATA + '/val/formatted/'
	# files = [f for f in os.listdir(val_loc) if '.csv' in f]
	# val_lst = argo_to_formatted(val_loc, output_dir, "val")
	# create_data(output_dir, val_lst, args.dir, "val")

	# test_loc = RAW_DATA + '/test_obs/data/'
	# output_dir = RAW_DATA + '/test_obs/formatted/'
	# files = [f for f in os.listdir(test_loc) if '.csv' in f]
	# test_lst = argo_to_formatted(test_loc, files, output_dir, "test")
	# create_data(output_dir, test_lst, args.dir, "test")

	print('using {} dataset.'.format(DATASET))

	t0 = time.time()

	model = TnpModel(viewArgs)
	if args.cuda:
		print("using cuda...\n")
	else:
		print("using cpu...\n")

	if LOAD != '':
		model.load(LOAD)
	t1 = time.time()

#	for i in range(5):
#		model.train(i)
	if TRAIN:
		model.train(0)

	t2 = time.time()

	if EVAL:
		model.evaluate()


	t3 = time.time()

	print('using {} dataset.'.format(DATASET))
	
	print('Loading time:{}'.format(t1 - t0))
	print("Training time:{}".format(t2 - t1))
	print("Testing time:{}".format(t3 - t2))
