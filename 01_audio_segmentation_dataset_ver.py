import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
#import librosa.display
import madmom
from pydub import AudioSegment
import math
import json

NEED_COPY_FOLDER = False
filename = 'piano.mp3'#'_original.mp3'
in_dir = 'D:/pop2jazz/'
#in_dir = '/Users/amandatsai/Downloads/pop2jazz/'
in_dir = os.path.join(in_dir, 'transcription_with_disjoint_notes.from_separated.soft_jazz_trio/20190403_095002.20190524_082516')
#source = os.path.join(in_dir, '_original.mp3')
out_dir='D:/piano_segments'
if not os.path.exists(out_dir): os.mkdir(out_dir)

folder_dict = {}
'''
def copy_folder_structure(inputpath, outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, os.path.relpath(dirpath, inputpath))
        if not os.path.isdir(structure): os.mkdir(structure)
        else: print("Folder does already exits!")
'''
def find_downbeats(source):
    proc = madmom.features.downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[4, 4], fps=100)
    act = madmom.features.downbeats.RNNDownBeatProcessor()(source)
    a = proc(act)
    #print(a[:,0])
    return a

def load_audio(source):
    sig, sr = librosa.core.load(source) # sr = 22050
    #sig = AudioSegment.from_file(source)
    #sound2 = AudioSegment.from_file(piano_file)
    #print('Time:{}, Length:{}'.format(len(sig)/sr, len(sig)))
    return sig, sr
    #return sig

def segment(source, outputpath, index, sr=1000):#, sr=44100):
    # Pydub do things in sr=1000
    a = find_downbeats(source)
    print('Track downbeats done ...')
    #sig = load_audio(source)
    sig, sr = load_audio(source)
    print('Load audio done ...')
    #sigs = []
    start = 0
    for i, c in enumerate(a):
        if c[-1] == 1:
            # if start == -1:
            #     start = 0
            #     continue
            # Output audio
            output_name = os.path.join(outputpath,'cut{:03d}_{:03d}.wav'.format(index, int(i/4)+1))
            print(output_name)
            #sigs.append(sig[start : int(i[0]*sr)])
            cut = sig[start : int(c[0]*sr)]
            #cut.export(output_name,format='wav')
            librosa.output.write_wav(output_name, cut, sr)
            start = int(c[0]*sr)
            #librosa.output.write_wav(output_name, sigs[i], sr)
    #sigs.append(sig[start:])
    #print(len(sigs)) # num of bars

def dictToTxt(in_dict, out_txt):
    with open(out_txt, 'w') as file:
        file.write(json.dumps(in_dict))

def txtToDict(in_txt):
    with open(in_txt, 'r') as file:
        a = json.load(file)
    return a
###########################################

# if NEED_COPY_FOLDER:
#     copy_folder_structure(in_dir, out_dir)
#     NEED_COPY_FOLDER = False

index = 0
for f in os.listdir(in_dir):
    if os.path.isdir(os.path.join(in_dir,f)):
        #print(f) #folder
        for ff in os.listdir(os.path.join(in_dir,f)):
            #print('---',ff) #sub-folder
            fder = os.path.join(os.path.join(in_dir,f),ff)
            if os.path.isdir(fder):
                # if index < 65:
                #     index += 1
                #     continue
                target_dir = os.path.join(out_dir, os.path.join(f,ff))
                print('> Target: ', target_dir)
                folder_dict['{:03d}'.format(index)] = target_dir.strip(out_dir)
                #print(folder_dict)
                save_dir = os.path.join(out_dir, '{:03d}'.format(index))
                if not os.path.isdir(save_dir): os.mkdir(save_dir)
                segment(os.path.join(fder, filename), save_dir, index)
                print('done')
                index += 1
                # for file in os.listdir(fder):
                #     print(file)

    dictToTxt(folder_dict, 'file.txt')

#d = txtToDict('file.txt')


