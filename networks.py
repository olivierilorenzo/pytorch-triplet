import torch.nn as nn
import torchvision.models


class EmbeddingResNet(nn.Module):
    def __init__(self):
        super(EmbeddingResNet, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        """
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(2048, 1000)
        """
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        for param in self.model.avgpool.parameters():
            param.requires_grad = True

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
