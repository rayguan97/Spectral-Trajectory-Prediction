import os
import numpy as np
from collections import defaultdict


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


def merge(file_names, output_dir, dtype, threadid):

    output_dir = output_dir + '/{}'
    traj = np.array([])

    track = defaultdict(dict)

    i = 0
    sz = len(file_names)
    for f in file_names:
        print("Start merging {}/{} in {} in thread {}...".format(i, sz, dtype, threadid))
        i += 1
        # print("Reading dataset {}...".format(d))
        npy_path = f
        # print(npy_path)
        data = np.load(npy_path, allow_pickle=True)

        # constructing train, val and testset for trajectory
        data0 = data[0]
        traj_id = np.unique(data0[:,1])
        d = int(data0[0, 0])


        if traj.size == 0:
            traj = data0
        else:
            traj = np.concatenate((traj, data0), axis=0)

        # constructing train, val and testset for tracks
        data1 = data[1]
        for ids in traj_id:
            track[d][ids] = data1[ids]
     

        # print("Dataset {} finsihed.".format(d))



    if not os.path.exists(output_dir.format(dtype)):
        os.makedirs(output_dir.format(dtype))

    # data for sgan
    sgan_name = "{}/{}Set{}.txt".format(dtype, dtype, str(threadid))
    f = open(output_dir.format(sgan_name), 'w')
    for line in traj:
        # f.write("{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4]))
        f.write("{}\t{}\t{}\t{}\t{}\n".format(int(line[2]), int(line[1]), line[3], line[4], int(line[0])))
    f.close()

    name = "{}/{}Set{}-traj.npy".format(dtype, dtype, str(threadid))
    np.save(output_dir.format(name), np.array([traj]))
    name = "{}/{}Set{}-track.npy".format(dtype, dtype, str(threadid))
    np.save(output_dir.format(name), np.array([track]))
    print("{} file in thread {} is saved and ready.".format(dtype, threadid))

    return len(traj)

input_dir = './resources/raw_data/ARGO/train/data/'
output_dir = './resources/raw_data/ARGO/train/formatted/'




