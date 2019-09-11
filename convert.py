import os
from pydub import AudioSegment
# files                                                    
base = 'D:/pop2jazz/transcription_with_disjoint_notes.from_separated.soft_jazz_trio/20190403_095002.20190524_082516/'
#dst = base

# convert wav to mp3
for dirPath, dirNames, fileNames in os.walk(base):
    for dd in dirNames:
        for sdirPath, sdirNames, sfileNames in os.walk(os.path.join(base, dd)):
            for sdd in sdirNames:
                for f in os.listdir(os.path.join(base, dd, sdd)):
                    if f.endswith('.mp3'):
                        src = os.path.join(base, dd, sdd, f)
                        print(src)
                        sound = AudioSegment.from_file(src, format='mp3')
                        # export to the same data path (only change the data format)
                        sound.export(src[:-3]+'wav', format="wav")