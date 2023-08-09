import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data_RainTrainL(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        target = cv2.imread(os.path.join(target_path,target_file))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        for j in range(2):
            input_file = "rain-%d.png" % (i + 1)
            input_img = cv2.imread(os.path.join(input_path,input_file))
            b, g, r = cv2.split(input_img)
            input_img = cv2.merge([r, g, b])

            target_img = target

            if j == 1:
                target_img = cv2.flip(target_img, 1)
                input_img = cv2.flip(input_img, 1)

            target_img = np.float32(normalize(target_img))
            target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)
            
            input_img = np.float32(normalize(input_img))
            input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

            print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))
            for n in range(target_patches.shape[3]):
                target_data = target_patches[:, :, :, n].copy()
                target_h5f.create_dataset(str(train_num), data=target_data)

                input_data = input_patches[:, :, :, n].copy()
                input_h5f.create_dataset(str(train_num), data=input_data)
                
                train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)
          
def prepare_data_RainTrainH(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path)
    target_path = os.path.join(data_path)

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        if os.path.exists(os.path.join(target_path,target_file)):

            target = cv2.imread(os.path.join(target_path,target_file))
            b, g, r = cv2.split(target)
            target = cv2.merge([r, g, b])

            input_file = "rain-%d.png" % (i + 1)

            if os.path.exists(os.path.join(input_path,input_file)): # we delete 546 samples

                input_img = cv2.imread(os.path.join(input_path,input_file))
                b, g, r = cv2.split(input_img)
                input_img = cv2.merge([r, g, b])

                target_img = target
                target_img = np.float32(normalize(target_img))
                target_patches = Im2Patch(target_img.transpose(2,0,1), win=patch_size, stride=stride)

                input_img = np.float32(normalize(input_img))
                input_patches = Im2Patch(input_img.transpose(2, 0, 1), win=patch_size, stride=stride)

                print("target file: %s # samples: %d" % (input_file, target_patches.shape[3]))

                for n in range(target_patches.shape[3]):
                    target_data = target_patches[:, :, :, n].copy()
                    target_h5f.create_dataset(str(train_num), data=target_data)

                    input_data = input_patches[:, :, :, n].copy()
                    input_h5f.create_dataset(str(train_num), data=input_data)

                    train_num += 1

    target_h5f.close()
    input_h5f.close()

    print('training set, # samples %d\n' % train_num)
          
def prepare_data_Rain100L(data_path, inputpath, targetpath):
    # train
    print('process test data')
    input_path = os.path.join(inputpath)
    target_path = os.path.join(targetpath)

    save_target_path = os.path.join(data_path, 'test_target.h5')
    save_input_path = os.path.join(data_path, 'test_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    test_num = 0
    for i in range(100):
        target_file = "norain (%d).png" % (i + 1)
        target_img = cv2.imread(os.path.join(target_path,target_file))
        #l = target_img.shape[0]
        #h = target_img.shape[1]
        #target_img = target_img[0:100, 0:100, :]
        b, g, r = cv2.split(target_img)
        target_img = cv2.merge([r, g, b])

        input_file = "rain (%d).png" % (i + 1)
        input_img = cv2.imread(os.path.join(input_path,input_file))
        #input_img = input_img[0:100, 0:100, :]
        b, g, r = cv2.split(input_img)
        input_img = cv2.merge([r, g, b])

        target_img = np.float32(normalize(target_img))
        #target_a = target_img.transpose(2,0,1)
        target_a = Im2Patch(target_img.transpose(2,0,1), win=100, stride=80)
        input_img = np.float32(normalize(input_img))
        #input_a = input_img.transpose(2,0,1)
        input_a = Im2Patch(input_img.transpose(2, 0, 1), win=100, stride=80)

        print("test file: %s" % (input_file))
        #target_a = target_a.reshape([3,320,320,1])
        #target_data = target_a[:, :, :].copy()
        #target_h5f.create_dataset(str(test_num), data=target_data)

        #input_a = input_a.reshape([3,320,320,1])
        #input_data = input_a[:, :, :].copy()
        #input_h5f.create_dataset(str(test_num), data=input_data)
        
        for n in range(target_a.shape[3]):
                target_data = target_a[:, :, :, n].copy()
                target_h5f.create_dataset(str(test_num), data=target_data)

                input_data = input_a[:, :, :, n].copy()
                input_h5f.create_dataset(str(test_num), data=input_data)

                test_num += 1

    target_h5f.close()
    input_h5f.close()

    print('testing set, # samples %d\n' % test_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)
    
class MyTestDataSet(udata.Dataset):
    def __init__(self, data_path='.'):
        super(MyTestDataSet, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'test_target.h5')
        input_path = os.path.join(self.data_path, 'test_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'test_target.h5')
        input_path = os.path.join(self.data_path, 'test_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()
        
        return torch.Tensor(input), torch.Tensor(target)