import numpy as np
import os
import json
import librosa
import random

test = 'test_'
path = test + 'music_segments/'
output_filepath_file = 'filenames'
output_label_file = 'labels' # song_index
output_triplets_file = 'triplets'
postfix = '.txt'

def gen_summary_file(wav_list, avg_file='avg.npy', std_file='std.npy'):
    s_list = []
    for w in wav_list:
        print(w)
        y, _ = librosa.core.load(os.path.join(path, w), sr=22050)
        S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
        s_list.append(S)
    avg = np.mean(np.array([np.mean(x, axis=1) for x in s_list]), axis=0)
    std = np.mean(np.array([np.std(x, axis=1) for x in s_list]), axis=0)
    np.save(avg_file, avg)
    np.save(std_file, std)
##########################################################################
def gen_wav_list():
    wav_list, label_list = [], []
    for dirPath, dirNames, fileNames in os.walk(path):
        for f in fileNames:
            if not f.endswith('.wav'): continue
            wav_list.append(os.path.join(dirPath.split('/')[-1], f))
    wav_list.sort()
    for w in wav_list: label_list.append(w[:w.rfind('/')])
    return wav_list, label_list

def gen_wav_list_split(split_num, index_num):
    assert split_num > 0 or index_num > 0
    wav_list, label_list = [], []
    indices = list(range(len(wav_list)))
    for dirPath, dirNames, fileNames in os.walk(path):
        for f in fileNames:
            if not f.endswith('.wav'): continue
            wav_list.append(os.path.join(dirPath.split('/')[-1], f))
    wav_list.sort()
    for w in wav_list: label_list.append(w[:w.rfind('/')])
    split_index = label_list.index('{:03d}'.format(split_num)) if split_num > 0 else index_num
    # if shuffle:
    #     #c = list(zip(wav_list, label_list))
    #     random.shuffle(indices)
    #     wav_shuf, label_shuf = [], []
    #     for i in indices:
    #         wav_shuf.append(wav_list[i])
    #         label_shuf.append(label_list[i])
    #     return indices, wav_shuf[:split_index], label_shuf[:split_index], 
    #             wav_shuf[split_index:], label_shuf[split_index:]
    return wav_list[:split_index], label_list[:split_index], wav_list[split_index:], label_list[split_index:]
##########################################################################
def make_triplet_list(wav_list, label_list):
    assert len(wav_list) == len(label_list)
    print('Processing Triplet Generation ...')
    ntriplets = len(wav_list)
    triplets = []
    for i in range(ntriplets-1):
        if not label_list[i+1] == label_list[i]: # change to another song
            #print('>>>', i)
            continue
        a = i
        b = np.random.choice([x for x in range(len(label_list)) if label_list[x] != label_list[i]], 1, replace=False)[0]
        #while np.any((a-b)==0): np.random.shuffle(b)
        c = i+1 
        triplets.append([a, b, c])
        #print(triplets)
    print('Done!')
    return triplets

def make_triplet_list_split(wav_train, label_train, wav_test, label_test):
    assert len(wav_train) == len(label_train)
    assert len(wav_test) == len(label_test)
    print('Processing Triplet Generation ...(train/test)')
    triplets = [list(), list()]
    name_list = ['Train', 'Test']
    wav_list = [wav_train, wav_test]
    label_list = [label_train, label_test]
    offset = [0, len(label_train)]
    print('Split_index = {}'.format(len(label_train)))

    for j in range(len(wav_list)):
        print('Triplet - {} dataset processing ...'.format(name_list[j]))
        ntriplets = len(wav_list[j])
        for i in range(ntriplets-1):
            if not label_list[j][i+1] == label_list[j][i]: # change to another song
                continue
            a = i + offset[j]
            b = i+1 + offset[j]
            cs = [x+offset[j] for x in range(len(label_list[j])) if label_list[j][x] != label_list[j][i]]
            c = np.random.choice(cs, 5, replace=False)
            
            print(wav_list[j][i])
            triplets[j].append([a, b, c])

    print('Done!')
    return triplets

def make_triplet_list_split_only_one(wav_train, label_train, wav_test, label_test):
    assert len(wav_train) == len(label_train)
    assert len(wav_test) == len(label_test)
    print('Processing Triplet Generation ...(train/test)')
    triplets = [list(), list()]
    name_list = ['Train', 'Test']
    wav_list = [wav_train, wav_test]
    label_list = [label_train, label_test]
    offset = [0, len(label_train)]
    print('Split_index = {}'.format(len(label_train)))

    for j in range(len(wav_list)):
        print('Triplet - {} dataset processing ...'.format(name_list[j]))
        ntriplets = len(wav_list[j])
        for i in range(ntriplets-1):
            if not label_list[j][i+1] == label_list[j][i]: # change to another song
                continue
            a = i + offset[j]
            b = i+1 + offset[j]
            cs = [x+offset[j] for x in range(len(label_list[j])) if x != i and x != i+1]#label_list[j][x] != label_list[j][i]]
            c = np.random.choice(cs, 5, replace=False)
            
            print(wav_list[j][i])
            triplets[j].append([a, b, c])

    print('Done!')
    return triplets
##########################################################################

def output_list_to_file(output_path_file, out_list):
    with open(output_path_file, 'w') as o:
        for l in out_list: o.write(l+'\n')

def output_triplets_to_file(output_path_file, out_lists):
    with open(output_path_file, 'w') as o:
        for l in out_lists:
            a, b, c = l
            cc = [int(x) for x in c]
            o.write('{} {} {}\n'.format(a,b,cc))
    # import csv
    # with open(output_path_file[:-4]+'___.txt', "w") as f:
    #     writer = csv.writer(f, delimiter=' ')
    #     writer.writerows(out_lists)


wav_list, label_list = gen_wav_list()
gen_summary_file(wav_list)
output_list_to_file(output_filepath_file+postfix, wav_list)
output_list_to_file(output_label_file+postfix, label_list)

split_num = 3#200
index_num = -1#300
wav_train, lab_train, wav_test, lab_test = gen_wav_list_split(split_num, index_num)

train, test = make_triplet_list_split(wav_train, lab_train, wav_test, lab_test)
#train, test = make_triplet_list_split_only_one(wav_train, lab_train, wav_test, lab_test)
output_triplets_to_file(output_triplets_file+'_train'+postfix, train)
output_triplets_to_file(output_triplets_file+'_test'+postfix, test)

#train = [[1,2,[2,3,4]],[3,4,[5,6,7]]]

#wav_list, label_list = gen_wav_list()
#triplets = make_triplet_list(wav_list, label_list)
#triplets = [[1,2,3],[4,5,6],[7,8,9]]
#output_list_to_file(output_filepath_file, wav_list)
#output_list_to_file(output_label_file, label_list)
#output_triplets_to_file(output_triplets_file, triplets)
#print(len(wav_list)) #22556
#print(len(triplets)) #22333



'''
import librosa
S_max = 0
data_path = 'audio_data/'
avgv = np.load(data_path + 'avg.npy')
stdv = np.load(data_path + 'std.npy')
def default_audio_loader(path, S_max):
    y, _ = librosa.core.load(path, sr=22050)
    S = librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
    if S.shape[-1] > S_max: S_max = S.shape[-1]
    print('MAX:',S_max)
    #S = np.transpose(np.log(1+10000*S))
    #S = (S-avgv)/stdv
    #S = np.expand_dims(S, 2)
    #print(S.shape)
    return S, S_max

for dirPath, dirNames, fileNames in os.walk(path):
    for f in fileNames:
        if not f.endswith('.wav'): continue
        s, S_max = default_audio_loader(os.path.join(dirPath, f), S_max)
print('>>MAX:',S_max)
'''
