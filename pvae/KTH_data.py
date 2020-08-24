import pickle
import imageio
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import rescale
from glob import glob
from natsort import natsorted
import torch



class KTHdataset(Dataset):
    def __init__(self, path):
        self.dict = self.get_data(path)
    def get_data(self, data_path):
        dirs = glob(os.path.join(data_path, '*'))
        dict = {}
        train_dict = {}
        test_dict = {}
        for dir in dirs:
            action = dir.split('/')[-1][:-3]
            video_folders = natsorted(glob(os.path.join(dir, '*')))
            train_vid_list = []
            test_vid_list = []
            for video_folder in video_folders:
                frames_list = []
                id = video_folder.split('/')[-1]
                person_number = [s for s in id if s.isdigit()]
                person_number = int(person_number[0] + person_number[1])
                frames = natsorted(glob(os.path.join(video_folder, '*.png')))
                for frame in frames:
                    frames_list.append(frame)
                if person_number <= 16:
                    train_vid_list.append(frames_list)
                else:
                    test_vid_list.append(frames_list)
            train_dict[action] = train_vid_list
            test_dict[action] = test_vid_list
        dict['train'] = train_dict
        dict['test'] = test_dict
        return dict
    def __getitem__(self, idx):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError



class SequenceKTHdataset(KTHdataset):
    def __init__(self, path, timesteps, movements='all', scale=0.5, mode ='train'):
        super(SequenceKTHdataset).__init__(path)
        self.timesteps = timesteps
        self.movements = movements
        self.sequences = self.create_sequences(self.dict[mode])
        # self.test_sequences = self.create_sequences(self.dict['test'])
        # self.stacked_test_sequences = self.flatten(self.test_sequences)
        self.stacked_sequences = self.flatten(self.sequences)
        self.scale = scale
    def __getitem__(self, idx):
        # rnd_action_index = np.random.randint(0, 6)
        # action = self.train_sequences[rnd_action_index]
        # rnd_seq_index = np.random.randint(len(action))
        sequence = self.stacked_sequences[idx]
        if self.scale == 1:
            img_stack = [imageio.imread(file)[..., 0:1] for file in sequence]
        else:
            img_stack = [rescale(imageio.imread(file)[..., 0:1], scale=self.scale, multichannel=True)for file in sequence]
        arr = np.stack(img_stack, axis=-1)
        arr = np.transpose(arr, (2, 0, 1, 3))

        if self.scale ==1 :
            arr = arr/255
        # arr = arr * 2 - 1

        return torch.Tensor(arr[:,:,:,0]), np.ones_like((arr.shape[0],1),'float64')
    def __len__(self):
        # arbitrary
        return  len(self.stacked_sequences)
    def create_sequences(self, dict):
        actions = []
        if self.movements == 'all':
            keylist = dict
        else:
            keylist = self.movements

        for key in keylist:
            sequences = []
            for video in dict[key]:
                for i in range(len(video) - self.timesteps):
                    sequences.append(video[i:i + self.timesteps])
            actions.append(sequences)
        return actions
    def flatten(self, actions):
        stacked_seqs = []
        for action in actions:
           for vid in action:
               stacked_seqs.append(vid)
        return stacked_seqs

    def get_test_sequence(self, idx):
        sequence = self.stacked_test_sequences[idx]
        scale = self.scale
        if scale == 1:
            img_stack = [imageio.imread(file)[..., 0:1] for file in sequence]
        else:
            img_stack = [rescale(imageio.imread(file)[..., 0:1], scale=scale, multichannel=True) for file in sequence]
        arr = np.stack(img_stack, axis=-1)
        arr = np.transpose(arr, (2, 0, 1, 3))
        if scale == 1:
            arr = arr/255
        arr = arr * 2 - 1
        arr = np.expand_dims(arr, 0)
        return arr


if __name__ == '__main__':
    movements = ['walking', 'handwaving']
    dataset = SequenceKTHdataset('../KTH_64', 1, movements=movements)
    for items in dataset:
        print(1)