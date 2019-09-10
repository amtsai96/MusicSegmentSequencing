#from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import librosa

data_path = '_sub_data/'#'_one_data/'#'audio_data/'
avgv = np.load(data_path + 'avg.npy')
stdv = np.load(data_path + 'std.npy')
def default_audio_loader(path, S_max, sr=22050):
    y, _ = librosa.core.load(path, sr=sr)
    #S = librosa.feature.chroma_stft(y=y, sr=sr)
    S = np.abs(librosa.stft(y, n_fft=4096))**2
    S = librosa.feature.chroma_stft(S=S, sr=sr)

    #print(S.shape)#(128, N)
    if S.shape[-1] > S_max: # clip
        S = S[:, :S_max]
    else:
        S = np.pad(S, ((0,0), (0, max(0, S_max-S.shape[-1]))),'constant', constant_values=(0))
    S = np.transpose(np.log(1+1000*S))
    #print(S.shape) #(S_max, 12)
    ##S = (S-avgv)/stdv
    #S = S / S.max()
    #print(S.max())
    #S = np.expand_dims(S, 2) # (N, 128, 1)
    #print(S.shape)
    return S

def _default_audio_loader(path, S_max, sr=22050):
    y, _ = librosa.core.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    #print(S.shape)#(128, N)
    if S.shape[-1] > S_max: # clip
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

class TripletAudioLoader(torch.utils.data.Dataset):
    S_max = 100
    filenames_filename = data_path + 'filenames.txt'
    train_triplet_file = data_path + 'triplets_train.txt'
    test_triplet_file = data_path + 'triplets_test.txt'
    #neg_num = 5

    def __init__(self, base_path, subfolder_path = 'test_music_segments', 
                transform=None, train=True,
                feature_extractor=default_audio_loader):
        """ filenames_filename: A text file with each line containing the path to an audio segment e.g.,
                music_segments/000/cut000-001.wav
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b [c d e f g]', a triplet is contained such that audio a is more 
                similar to audio b than it is to audio c, d, e, f,and g. e.g., 41 42 [2000 123 547 47 99]
                (Since we define positive case is exactly the next segment)
        """
        self.base_path = base_path 
        self.subfolder = subfolder_path
        self.train = train
        #ancs, poss, negs = [], [], [] # Anchor, Positive, Negative

        if self.train: FILE_OPEN = self.train_triplet_file
        else: FILE_OPEN = self.test_triplet_file
        with open(os.path.join(base_path, FILE_OPEN)) as f:
            triplets = []
            for line in f:
                #ancs.append(int(line.split()[0]))
                #poss.append(int(line.split()[1]))
                negs = line[line.rfind('[')+1 : line.rfind(']')].split(', ')
                for i in range(len(negs)):
                    triplets.append((int(line.split()[0]), int(line.split()[1]), int(negs[i]))) # anchor, close, far
                #negs = [int(a) for a in negs]
                #triplets.append((int(line.split()[0]), int(line.split()[1]), negs)) # anchor, close, far
        self.triplets = triplets#(ancs, poss, negs)

        if self.train:
            self.triplets_train = self.triplets
        else:
            self.triplets_test = self.triplets

        self.audio_path = []
        with open(self.filenames_filename) as f:
            for line in f:
                self.audio_path.append(os.path.join(self.base_path, self.subfolder, line.rstrip('\n')))
        self.transform = transform
        self.feature_extractor = feature_extractor

        #self.ancs = ancs # Anchor
        #self.poss = poss # Positive
        #self.negs = negs # Negative
        #self.label = label # from which song

    # def __getitem__(self, index):
    #     path1, path2, path3 = self.triplets[index]
    #     #path3 = path3s[0] # we have more than one "far samples" ... 
    #     img1 = self.feature_extractor(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path1)]), self.S_max)
    #     img2 = self.feature_extractor(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path2)]), self.S_max)
    #     img3 = self.feature_extractor(os.path.join(self.base_path, self.subfolder, self.filenamelist[int(path3)]), self.S_max)
    #     if self.transform is not None:
    #         img1 = self.transform(img1)
    #         img2 = self.transform(img2)
    #         img3 = self.transform(img3)
    #     return img1, img2, img3

    def __getitem__(self, index):
        data = []
        for i in range(len(self.triplets[index])):
            data.append(self.feature_extractor(self.audio_path[int(self.triplets[index][i])], self.S_max))
            if self.transform is not None:
                data[i] = self.transform(data[i])
        return data

    # def __getitem__(self, index):
    #     assert len(self.triplets[index]) == 3
    #     data = []
    #     for i in range(len(self.triplets[index])):
    #         if i < 2: # Anchor, Pos
    #             data.append(self.feature_extractor(self.audio_path[int(self.triplets[index][i])], self.S_max))
    #             if self.transform is not None:
    #                 data[i] = self.transform(data[i])
    #         else:
    #             negs = []
    #             for j in range(self.neg_num):
    #                 negs.append(self.feature_extractor(self.audio_path[int(self.triplets[index][i][j])], self.S_max))
    #             if self.transform is not None:
    #                 negs[i] = self.transform(negs[i])
    #             data.append(negs)
    #     #print(len(data), len(data[2])) #(3, 5)
    #     return data

    def __len__(self):
        return len(self.triplets)
