import numpy as np
def summary(ndarr):
    #print(ndarr)
    print('* shape: {}'.format(ndarr.shape))
    print('* min: {}'.format(np.min(ndarr)))
    print('* max: {}'.format(np.max(ndarr)))
    print('* avg: {}'.format(np.mean(ndarr)))
    print('* std: {}'.format(np.std(ndarr)))
    #print('* unique: {}'.format(np.unique(ndarr)))
a = np.load('std.npy')
summary(a)
#print(a.max())
print('---------------')
b = np.load('avg.npy')
summary(b)
#print(b.max()) # 10.825