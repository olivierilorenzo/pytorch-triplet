import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class EmbeddingAlexNet(nn.Module):
    def __init__(self, mode):
        super(EmbeddingAlexNet, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.alexnet(pretrained=False)
        else:
            if mode == 'ft':
                self.model = torchvision.models.alexnet(pretrained=True)

                for param in self.model.parameters():
                    param.requires_grad = False

                for i in range(8, 13):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
                for param in self.model.avgpool.parameters():
                    param.requires_grad = True
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)

    def classification_setup(self, n_classes):
        self.model.classifier[6] = nn.Linear(4096, n_classes)


class EmbeddingVgg16(nn.Module):
    def __init__(self, mode):
        super(EmbeddingVgg16, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.vgg16_bn(pretrained=False)
        else:
            if mode == 'ft':
                self.model = torchvision.models.vgg16_bn(pretrained=True)
                for param in self.model.parameters():
                    param.requires_grad = False

                for i in range(30, 44):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
                for param in self.model.avgpool.parameters():
                    param.requires_grad = True
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)

    def classification_setup(self, n_classes):
        self.model.classifier[6] = nn.Linear(4096, n_classes)


class EmbeddingDenseNet(nn.Module):
    def __init__(self, mode):
        super(EmbeddingDenseNet, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.densenet121(pretrained=False)
        else:
            if mode == 'ft':
                self.model = torchvision.models.densenet121(pretrained=True)
                for param in self.model.parameters():
                    param.requires_grad = False

                for param in self.model.features.denseblock4.parameters():
                    param.requires_grad = True
                for param in self.model.features.norm5.parameters():
                    param.requires_grad = True
                for param in self.model.classifier.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)

    def classification_setup(self, n_classes):
        self.model.classifier = nn.Linear(1024, n_classes)


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

    def classification_setup(self, n_classes):
        self.model.fc = nn.Linear(2048, n_classes)


class EmbeddingResNeXt(nn.Module):
    def __init__(self, mode):
        super(EmbeddingResNeXt, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.resnext50_32x4d(pretrained=False)
        else:
            self.model = torchvision.models.resnext50_32x4d(pretrained=True)

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

    def classification_setup(self, n_classes):
        self.model.fc = nn.Linear(2048, n_classes)


class EmbeddingGoogleNet(nn.Module):
    def __init__(self, mode):
        super(EmbeddingGoogleNet, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.googlenet(pretrained=False)
        else:
            if mode == 'ft':
                self.model = torchvision.models.googlenet(pretrained=True)
                for param in self.model.parameters():
                    param.requires_grad = False

                for param in self.model.inception5b.parameters():
                    param.requires_grad = True
                for param in self.model.avgpool.parameters():
                    param.requires_grad = True
                for param in self.model.dropout.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)

    def classification_setup(self, n_classes):
        self.model.fc = nn.Linear(1024, n_classes)


class EmbeddingInception(nn.Module):
    """
    Inception object for full network training or fine-tuning on a specific layer
    """
    def __init__(self, mode):
        super(EmbeddingInception, self).__init__()
        if mode == 'full':
            self.model = torchvision.models.inception_v3(pretrained=False)
        else:
            if mode == 'ft':
                self.model = torchvision.models.inception_v3(pretrained=True)

                for param in self.model.parameters():
                    param.requires_grad = False

                for param in self.model.Mixed_7a.parameters():
                    param.requires_grad = True

                for param in self.model.Mixed_7b.parameters():
                    param.requires_grad = True

                for param in self.model.Mixed_7c.parameters():
                    param.requires_grad = True

                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_embedding(self, x):
        return self.forward(x)

    def classification_setup(self, n_classes):
        self.model.fc = nn.Linear(2048, n_classes)
        self.model.AuxLogits.fc = nn.Linear(768, n_classes)
