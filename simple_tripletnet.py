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
        dists = [F.pairwise_distance(embedded_a, embedded_p, 2)]
        dists.append(F.pairwise_distance(embedded_a, embedded_n, 2))
        return dists[0], dists[1], embedded_a, embedded_p, embedded_n

    def get_embedding(self, x):
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(100, 70)
        self.fc2 = nn.Linear(70, 40)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)
