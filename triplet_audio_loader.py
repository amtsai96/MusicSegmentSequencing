#from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import librosa

data_path = '_sub_data/'#'_one_data/'#'audio_data/'
# def default_image_loader(path):
#     return Image.open(path).convert('RGB')

avgv = np.load(data_path + 'avg.npy')
stdv = np.load(data_path + 'std.npy')

def default_audio_loader(path, S_max):
    y, _ = librosa.core.load(path, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    #print(S.shape)#(128, N)
    if S.shape[-1] > S_max:
        S = S[:, :S_max]
        #print(S.shape)
    else:
        S = np.pad(S, ((0,0), (0, max(0, S_max-S.shape[-1]))),'constant', constant_values=(0))
    #print(S.shape) #(128, S_max)
    #S = np.transpose(np.log(1+10000*S))
    S = np.transpose(np.log(1+1000*S))
    #print(S.shape) #(S_max, 128)
    S = (S-avgv)/stdv
    #S = np.expand_dims(S, 2) # (N, 128, 1)
    #print(S.shape)
    return S

# def summary(ndarr):
#     print(ndarr)
#     print('* shape: {}'.format(ndarr.shape))
#     print('* min: {}'.format(np.min(ndarr)))
#     print('* max: {}'.format(np.max(ndarr)))
#     print('* avg: {}'.format(np.mean(ndarr)))
#     print('* std: {}'.format(np.std(ndarr)))
#     print('* unique: {}'.format(np.unique(ndarr)))

class TripletAudioLoader(torch.utils.data.Dataset):
    #training_file = 'training.pt'
    #test_file = 'test.pt'
    S_max = 100
    filenames_filename = data_path + 'filenames.txt'
    train_triplet_file = data_path + 'triplets_train.txt'
    test_triplet_file = data_path + 'triplets_test.txt'
    def __init__(self, base_path, subfolder_path = 'test_music_segments', 
                transform=None, train=True,
                loader=default_audio_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image b than it is to image c, e.g., 0 42 2000
        """
        self.base_path = base_path 
        self.subfolder = subfolder_path
        self.train = train 
        if self.train:
            # self.train_data, self.train_labels = torch.load(
            #     os.path.join(base_path, self.training_file))
            # self.make_triplet_list(n_train_triplets)
            triplets = []
            for line in open(os.path.join(base_path, self.train_triplet_file)):
                negs = line[line.rfind('[')+1 : line.rfind(']')].split(', ')
                for i in range(len(negs)):
                    triplets.append((int(line.split()[0]), int(line.split()[1]), int(negs[i]))) # anchor, close, far
                #triplets.append((int(line.split()[0]), int(line.split()[1]), int(line.split()[2]))) # anchor, close, far
            self.triplets_train = triplets
        else:
            triplets = []
            for line in open(os.path.join(base_path, self.test_triplet_file)):
                negs = line[line.rfind('[')+1 : line.rfind(']')].split(', ')
                for i in range(len(negs)):
                    triplets.append((int(line.split()[0]), int(line.split()[1]), int(negs[i]))) # anchor, close, far
                #triplets.append((int(line.split()[0]), int(line.split()[1]), int(negs[0]))) # anchor, close, far
            self.triplets_test = triplets

        self.filenamelist = []
        for line in open(self.filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        #path3 = path3s[0] # we have more than one "far samples" ... 
        img1 = self.loader(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path1)]), self.S_max)
        img2 = self.loader(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path2)]), self.S_max)
        img3 = self.loader(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path3)]), self.S_max)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)
