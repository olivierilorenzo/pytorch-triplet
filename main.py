import torch
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from datasets import offline_data_aug
from datasets import TVReID
from datasets import BalancedBatchSampler
from networks import EmbeddingResNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from evaluation import evaluate

# data augmentation
transform_list = [transforms.RandomHorizontalFlip(1), transforms.RandomCrop(224, 64)]
offline_data_aug(transform_list,300,200)

# preparazione dataset
train_dataset = TVReID(train=True, pid_max=300, pid_min=200)
test_dataset = TVReID(train=False, pid_max=310, pid_min=200, non_target=10)

# preparazione modello
cuda = torch.cuda.is_available()

train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=40)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=40)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

margin = 1.

model = EmbeddingResNet()

if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

# restart da checkpoint
"""
checkpoint = torch.load('checkpoint.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
"""

fit(online_train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AverageNonzeroTripletsMetric()])

evaluate(train_dataset, test_dataset, model, thresh=3.4)
