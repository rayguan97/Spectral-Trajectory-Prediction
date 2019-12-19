import os
import numpy as np


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



input_dir = RAW_DATA + './resources/raw_data/ARGO/train/data/'
output_dir = RAW_DATA + './resources/raw_data/ARGO/train/formatted/'




