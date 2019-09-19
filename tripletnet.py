#import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, A, P, N):
        embedded_a = self.embedding_net(A.float())
        embedded_p = self.embedding_net(P.float())
        embedded_n = self.embedding_net(N.float())
        dist_ap = F.pairwise_distance(embedded_a, embedded_p, 2)
        dist_an = F.pairwise_distance(embedded_a, embedded_n, 2)
        return dist_ap, dist_an, embedded_a, embedded_p, embedded_n

    def get_embedding(self, x):
        return self.embedding_net(x)

# Chromagram - Conv1D
class ChromagramEmbeddingNet(nn.Module):
    def __init__(self):
        super(ChromagramEmbeddingNet, self).__init__()
        FEATURE_DIM = 12
        self.conv1 = nn.Conv1d(FEATURE_DIM, 20, kernel_size=3)
        self.conv2 = nn.Conv1d(20, 50, kernel_size=3)
        self.conv2_drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(100, 70)
        self.fc2 = nn.Linear(70, 50)

    def forward(self, x):
        x = x.view(-1, x.size(2), x.size(3))
        x = x.transpose(1, 2)
        #print(x.shape)#(BATCH_SIZE, IN_DIM, S_MAX)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        #print(x.shape)#(BATCH_SIZE, CONV1_OUT, )
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)#(BATCH_SIZE, CONV2_OUT, )
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

# Mel-Spectrogram - Conv1D
class MelSpectrogramEmbeddingNet(nn.Module):
    def __init__(self):
        super(MelSpectrogramEmbeddingNet, self).__init__()
        FEATURE_DIM = 128
        self.conv1 = nn.Conv1d(FEATURE_DIM, 70, kernel_size=3)
        self.conv2 = nn.Conv1d(70, 100, kernel_size=3)
        self.conv2_drop = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(100, 70)
        self.fc2 = nn.Linear(70, 50)

    def forward(self, x):
        #print(x.shape) #(BATCH_SIZE, IN_CHANNEL, S_MAX, IN_DIM)
        x = x.view(-1, x.size(2), x.size(3))
        x = x.transpose(1, 2)
        #print(x.shape)#(BATCH_SIZE, IN_DIM, S_MAX)
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        #print(x.shape)#(BATCH_SIZE, CONV1_OUT, )
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)#(BATCH_SIZE, CONV2_OUT, )
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

# Mel-Spectrogram - Conv2D
class MelSpectrogram2DEmbeddingNet(nn.Module):
    def __init__(self):
        super(MelSpectrogram2DEmbeddingNet, self).__init__()
        FEATURE_DIM = 128
        self.conv1 = nn.Conv2d(FEATURE_DIM, 50, kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(50, 100, kernel_size=(3, 1))
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(100, 70)
        self.fc2 = nn.Linear(70, 50)

    def forward(self, x):
        #print(x.shape)#(BATCH_SIZE, IN_CHANNEL, S_MAX, FEATURE_DIM)
        x = x.transpose(1, 3)
        #print(x.shape)#(BATCH_SIZE, FEATURE_DIM, S_MAX, IN_CHANNEL)
        x = F.relu(F.max_pool2d(self.conv1(x), (2, 1)))
        #print(x.shape)#(BATCH_SIZE, CONV2_IN, , )
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), (2, 1)))
        #print(x.shape)#(BATCH_SIZE, FC1_IN, , )
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)

# class MelSpectrogram2DEmbeddingNet(nn.Module):
#     def __init__(self):
#         super(MelSpectrogram2DEmbeddingNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(100, 70)
#         self.fc2 = nn.Linear(70, 50)

#     def forward(self, x):
#         #print(x.shape)#(BATCH_SIZE, CONV1_IN, S_MAX, FEATURE_DIM)
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         #print(x.shape)#(BATCH_SIZE, CONV2_IN, , )
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         #print(x.shape)#(BATCH_SIZE, FC1_IN, , )
#         x = x.view(-1, 100)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         return self.fc2(x)


