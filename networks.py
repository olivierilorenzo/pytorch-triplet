import torch.nn as nn
import torchvision.models


class EmbeddingResNet(nn.Module):
    """
    ResNet50 object for full network training or fine-tuning on a specific layer
    """
    def __init__(self, mode):
        super(EmbeddingResNet, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.resnet50(pretrained=False)
        else:
            self.model = torchvision.models.resnet50(pretrained=True)

            for param in self.model.parameters():
                param.requires_grad = False

            if mode == 'layer1':
                for param in self.model.layer1.parameters():
                    param.requires_grad = True

                for param in self.model.layer2.parameters():
                    param.requires_grad = True

                for param in self.model.layer3.parameters():
                    param.requires_grad = True

                for param in self.model.layer4.parameters():
                    param.requires_grad = True

                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

            if mode == 'layer2':
                for param in self.model.layer2.parameters():
                    param.requires_grad = True

                for param in self.model.layer3.parameters():
                    param.requires_grad = True

                for param in self.model.layer4.parameters():
                    param.requires_grad = True

                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

            if mode == 'layer3':
                for param in self.model.layer3.parameters():
                    param.requires_grad = True

                for param in self.model.layer4.parameters():
                    param.requires_grad = True

                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

            if mode == 'layer4':
                for param in self.model.layer4.parameters():
                    param.requires_grad = True

                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

            if mode == 'avgpool':
                for param in self.model.avgpool.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

            if mode == 'fc':
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingVgg16(nn.Module):
    def __init__(self, mode):
        super(EmbeddingVgg16, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.vgg16_bn(pretrained=False)
        else:
            self.model = torchvision.models.vgg16_bn(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    """
    Usata con la generazione random delle triplette, poco efficiente
    """
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
