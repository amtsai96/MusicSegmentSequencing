import os
import os.path
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import librosa

data_path = 'audio_data/'
music_folder = './test_piano_segments/'
filenames_txt = data_path + 'filenames.txt'
avgv = np.load(data_path + 'avg.npy')
stdv = np.load(data_path + 'std.npy')


def audio_chromagram_loader(path, S_max, sr=22050):
    y, _ = librosa.core.load(path, sr=sr)
    S = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    #S = np.abs(librosa.stft(y, n_fft=4096))**2
    #S = librosa.feature.chroma_stft(S=S, sr=sr)
    if S.shape[-1] > S_max: S = S[:, :S_max]
    else:
        S = np.pad(S, ((0, 0), (0, max(0, S_max-S.shape[-1]))), 'constant', constant_values=(0))
    S = np.transpose(S)
    #print(S.shape) #(S_max, 12)
    return S


def audio_mel_spectrogram_loader(path, S_max, sr=22050):
    y, _ = librosa.core.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    # print(S.shape)#(128, N)
    if S.shape[-1] > S_max: S = S[:, :S_max]
    else:
        S = np.pad(S, ((0, 0), (0, max(0, S_max-S.shape[-1]))), 'constant', constant_values=(0))
    # print(S.shape) #(128, S_max)
    #S = np.transpose()
    S = np.transpose(np.log(1+1000*S))
    # print(S.shape) #(S_max, 128)
    S = (S-avgv)/stdv
    # S = np.expand_dims(S, 2) # (N, 128, 1)
    # print(S.shape)
    return S


class TripletAudioLoader(torch.utils.data.Dataset):
    S_max = 100
    #neg_num = 5
    def __init__(self, data_txt, feature, 
                audio_file_folder=music_folder, transform=None):
                 #feature_extractor=audio_mel_spectrogram_loader):
                 #feature_extractor=audio_chromagram_loader):
        """ filenames_txt: A text file with each line containing the path to an audio segment e.g.,
                music_segments/000/cut000-001.wav
            triplets_file_name: A text file with each line containing two integers and one list, 
                where integer i refers to the i-th segment in the filenames file. 
                For a line of intergers 'a b [c d e f g]', a triplet is contained such that audio a is more 
                similar to audio b than it is to audio c, d, e, f,and g. e.g., 41 42 [20 123 547 47 99]
                (Since we define the positive case is exactly the next segment)
        """
        # ancs, poss, negs = [], [], [] # Anchor, Positive, Negative
        with open(os.path.join(data_path, data_txt)) as f:
            triplets = []
            for line in f:
                # ancs.append(int(line.split()[0]))
                # poss.append(int(line.split()[1]))
                negs = line[line.rfind('[')+1: line.rfind(']')].split(', ')
                for i in range(len(negs)):
                    # anchor, close, far
                    triplets.append((int(line.split()[0]), int(line.split()[1]), int(negs[i])))
                #negs = [int(a) for a in negs]
                # triplets.append((int(line.split()[0]), int(line.split()[1]), negs)) # anchor, close, far
        self.triplets = triplets  # (ancs, poss, negs)
        self.audio_file_folder = audio_file_folder
        self.audio_path = []
        with open(filenames_txt) as f:
            for line in f:
                self.audio_path.append(os.path.join(self.audio_file_folder, line.rstrip('\n')))
        self.transform = transform
        self.feature_extractor = audio_mel_spectrogram_loader if feature == 'mel_spec' else audio_chromagram_loader

        # self.ancs = ancs # Anchor
        # self.poss = poss # Positive
        # self.negs = negs # Negative
        # self.label = label # from which song

    def __getitem__(self, index):
        data = []
        for i in range(len(self.triplets[index])):
            d = self.feature_extractor(self.audio_path[int(self.triplets[index][i])], self.S_max)
            data.append(d)
            if self.transform is not None:
                data[i] = self.transform(data[i])
        return data

    # def __getitem__(self, index):
    #     assert len(self.triplets[index]) == 3
    #     data = []
    #     for i in range(len(self.triplets[index])):
    #         if i < 2: # Anchor, Pos
    #             data.append(self.feature_extractor(self.audio_path[int(self.triplets[index][i])], self.S_max))
    #             if self.transform is not None: data[i] = self.transform(data[i])
    #         else:
    #             negs = []
    #             for j in range(self.neg_num):
    #                 negs.append(self.feature_extractor(self.audio_path[int(self.triplets[index][i][j])], self.S_max))
    #             if self.transform is not None: negs[i] = self.transform(negs[i])
    #             data.append(negs)
    #     #print(len(data), len(data[2])) #(3, 5)
    #     return data

    def __len__(self):
        return len(self.triplets)
