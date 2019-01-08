import os
import numpy as np 
import math
import shutil

def random_distrib(original_path, new_path):
    for dir_item in os.listdir(original_path):

        original_file_list = []
        new_file_path = []
        label = []

        original_sub_path = os.path.join(original_path, dir_item)
        new_sub_path = os.path.join(new_path, dir_item)
        if not os.path.exists(new_sub_path):
            os.makedirs(new_sub_path)
        for file_item in os.listdir(original_sub_path):
            temp_originalfile_path = os.path.join(original_sub_path, file_item)
            temp_newfile_path = os.path.join(new_sub_path, file_item)
            original_file_list.append(temp_originalfile_path)
            new_file_path.append(temp_newfile_path)
            label.append(dir_item)

        path_list = np.array([original_file_list, new_file_path, label])
        path_list = path_list.transpose()
        np.random.shuffle(path_list)
        
        from_path = list(path_list[ : , 0])
        to_path = list(path_list[ : , 1])
        n_file = len(from_path)
        n_test = int(math.ceil(n_file * 0.3))

        for i in range(n_test):
            shutil.move(from_path[i], to_path[i])
        '''
        for i in path_list:
            print(i)
        '''
        # print(path_list.shape)
        # return

    # concatenate连接
    # path_list = np.hstack((original_file_list, new_file_path, label))


if __name__ == "__main__":
    if not os.path.exists("img_test"):
        os.mkdir("img_test")
    random_distrib('img_train/', 'img_test/')
