import os
import numpy as np

def generate_data_from_txt(files_path):
    '''
    to convert the data from all txt files into one large numpy array
    :param files_path: filepath to load
    :return: numpy array of the data from txt file
    '''
    data = np.loadtxt(files_path)

    return data

def save_to_file(data,files_path_to_sv):
    '''
    to save the obtained merged numpy array to .npy
    :param data: large numpy array
    :param files_path_to_sv: file path where this .npy file will be saved
    :return: None
    '''
    np.save(files_path_to_sv, data)

def load_data(files_path_saved):
    '''
    to load the numpy array from .npy file
    :param files_path_to_sv: file path where the numpy array is saved
    :return: loaded numpy array
    '''
    data = np.load(files_path_saved)
    return data



def first_row_process(data):
    '''
    to format the first row, i.e., object ID to start from 0 to maximum
    :param data: merged numpy array
    :return: formatted numpy array
    '''
    data1 = data

    # max_val = (np.amax(data[:,1]))
    min_val = (np.amin(data1[:,1]))

    n1 = min_val * np.ones(data1.shape[0], dtype = int)

    data1[:,1] = data1[:,1] - n1

    return data1

def zero_row_process(zero_row):
    '''
    to format the zeroth row, i.e., all the time stamps are converted to 1 to maximum scale
    :param zero_row: the zeroth row of the merged numpy array
    :return: formatted zeroth row of the merged numpy array, that is the frame ID's
    '''

    first_ele = zero_row[0]
    frame_ID_list = []
    j = 1

    for i in zero_row:
        if i == first_ele:
            frame_ID_list.append(j)
        else:
            first_ele = i
            j+=1
            frame_ID_list.append(j)

    frame_ID_array = np.asarray(frame_ID_list, dtype= int).T

    return frame_ID_array


def save_to_text(final, to_save_txt, index):
    '''
    save the formatted array to a .txt file
    :param final: final formatted array
    :param to_save_txt: file path where the .txt file has rto be saved
    :param index: index value which will be used as a dataset ID
    :return: None
    '''
    ind = index

    lisss = np.ndarray.tolist(final)
    for items in lisss:
        items[0] = int(items[0])
        items[1] = int(items[1])

    with open(to_save_txt, 'w') as filehandle:
        for l in lisss:
            # filehandle.write('%d \t %d \t %f \t %f \t %f \n' %(l[0], l[1], l[2], l[3], l[4]))
            filehandle.write("{},{},{},{},{}\n".format(ind,l[1],l[0],l[2],l[3]))

def apolloscape_to_formatted_test(dir, train_dir):
    '''
    to create the formatted data from the apolloscape data
    :param dir: directory of the apolloscape dataset
    :return: None. Created formatted .txt files
    '''

    files_path = dir

    file_names = []

    for file in sorted(os.listdir(files_path)):
        if file.endswith('.txt'):
            file_names.append(file)


    index = 1

    for file in file_names:
        filepath = files_path + file

        data = generate_data_from_txt(filepath)

        data = first_row_process(data)

        zero_row = data[:,0]

        corrected_zero_row = zero_row_process(zero_row)
        data[:,0] = corrected_zero_row

        formatted_data = np.delete(data,[2,5,6,7,8,9],axis=1)
        to_save_txt = train_dir + 'APOL/test_obs/traj{:>04}.txt'.format(index)

        save_to_text(formatted_data, to_save_txt, index)

        index+=1
        # break

def apolloscape_to_formatted(dir, dir_test):
    '''
    to create the formatted data from the apolloscape data
    :param dir: directory of the apolloscape dataset
    :return: None. Created formatted .txt files
    '''

    files_path = dir

    file_names = []

    for file in sorted(os.listdir(files_path)):
        if file.endswith('.txt'):
            file_names.append(file)

    index = 1

    for file in file_names:
        filepath = files_path + file


        data = generate_data_from_txt(filepath)

        data = first_row_process(data)

        zero_row = data[:,0]
        corrected_zero_row = zero_row_process(zero_row)
        data[:,0] = corrected_zero_row
        formatted_data = np.delete(data,[4,5,6,7,8],axis=1)
        # n_ones = np.ones((formatted_data.shape[0],1))
        # final = np.concatenate((formatted_data, n_ones), axis=1)
        # print(final[:,0])

        if index <= 8:
            to_save_txt = files_path + 'APOL/train/traj{:>04}.txt'.format(index)
            print('index in if<=8',index)

        else:
            to_save_txt = files_path + 'APOL/val/traj{:>04}.txt'.format(index)
            print('index in if = 9', index)

        save_to_text(formatted_data, to_save_txt, index)

        index+=1
        # break
    apolloscape_to_formatted_test(dir_test, dir)



'''
Instructions for directory structure:

1. Download the dataset from the link provided in the README.md
2. Unzip the downloaded files and
3. Follow below format  
DATA_DIR = folder_where_unzipped_apolloscape_data_is_present + '/sample_trajectory/asdt_sample_ trajectory/'
DATA_DIR_TEST = folder_where_unzipped_apolloscape_data_is_present + '/prediction_test/'
'''

apolloscape_to_formatted(DATA_DIR, DATA_DIR_TEST)
