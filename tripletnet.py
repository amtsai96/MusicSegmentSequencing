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
FEATURE_DIM = 12
class ChromagramEmbeddingNet(nn.Module):
    def __init__(self):
        super(ChromagramEmbeddingNet, self).__init__()
        self.conv1 = nn.Conv1d(FEATURE_DIM, 100, kernel_size=3)
        self.conv2 = nn.Conv1d(100, 50, kernel_size=3)
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
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


# Mel-Spectrogram - Conv2D
class MelSpectrogramEmbeddingNet(nn.Module):
    def __init__(self):
        super(MelSpectrogramEmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(100, 70)
        self.fc2 = nn.Linear(70, 50)

    def forward(self, x):
        #print(x.shape)#(BATCH_SIZE, CONV1_IN, S_MAX, FEATURE_DIM)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #print(x.shape)#(BATCH_SIZE, CONV2_IN, , )
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print(x.shape)#(BATCH_SIZE, FC1_IN, , )
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #print(x.shape)#(N, FC2_IN)
        return self.fc2(x)
