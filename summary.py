import numpy as np
import os
import librosa

def summary(ndarr):
    print(ndarr)
    print('* shape: {}'.format(ndarr.shape))
    print('* min: {}'.format(np.min(ndarr)))
    print('* max: {}'.format(np.max(ndarr)))
    print('* avg: {}'.format(np.mean(ndarr)))
    print('* std: {}'.format(np.std(ndarr)))
    print('* unique: {}'.format(np.unique(ndarr)))

def default_audio_loader(path):
    print(path)
    y, _ = librosa.core.load(path, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    #S = np.transpose(np.log(1+10000*S))
    #S = (S-avgv)/stdv
    #S = np.expand_dims(S, 2)
    return S

def info(ndarr):
    print('* avg: {}'.format(np.mean(ndarr, axis=1)))
    print('* std: {}'.format(np.std(ndarr, axis=1)))
    
s = []
path='/Users/amandatsai/FILES/NCKU/Academia_Sinica/test/MASHUP/music_segments/'
for dirPath, dirNames, fileNames in os.walk(path):
    for f in fileNames:
        if f.endswith('.wav'):
            s.append(default_audio_loader(os.path.join(path,os.path.join(dirPath, f))))
'''
s = [default_audio_loader(path+a) for a in os.listdir(path) if a.endswith('.wav')]
#summary(s)
# a = default_audio_loader(path+'cut000_001.wav')
# b = default_audio_loader(path+'cut000_002.wav')
# s = [a, b]
# print(a.shape)
# b=np.mean(a, axis=1)
# print(b, b.shape)
'''
avg = np.mean(np.array([np.mean(x, axis=1) for x in s]), axis=0)
# avg = np.mean(avgs, axis=0)
print(avg.shape)
std = np.std(np.array([np.std(x, axis=1) for x in s]), axis=0)
print(std.shape)

np.save('avg.npy', avg)
np.save('std.npy', std)
