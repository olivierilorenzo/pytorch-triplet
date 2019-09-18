import torch
import argparse
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
from datasets import offline_data_aug
from datasets import TVReID
from datasets import BalancedBatchSampler
from networks import EmbeddingResNet, EmbeddingVgg16, EmbeddingInception, EmbeddingAlexNet, EmbeddingResNeXt, EmbeddingDenseNet, EmbeddingGoogleNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from evaluation import evaluate, evaluate_gpu, evaluate_vram_opt, classification

parser = argparse.ArgumentParser()
parser.add_argument('--train_min', default=0, help="first pid of the training interval", type=int)
parser.add_argument('--train_max', help="last pid of the training interval", type=int)
parser.add_argument('--test_min', default=0, help="first pid of the testing interval", type=int)
parser.add_argument('--test_max', help="last pid of the testing interval", type=int)
parser.add_argument('--data_aug', default=0, help="<0> no data augmentation, <1> horizontal flip + random crop, "
                                                  "<2> horizontal flip + vertical flip + random crop", type=int)
parser.add_argument('--non_target', default=0, help="n of impostors", type=int)
parser.add_argument('--classes', default=5, help="n of classes in the mini-batch", type=int)
parser.add_argument('--samples', default=20, help="n of sample per class in the mini-batch", type=int)
parser.add_argument('--network', default="resnet", help="choose between <resnet>, <vgg16>, <alexnet>, <densenet>, "
                                                        "<resnext>, <googlenet> and <inception>(v3) "
                                                        "for feature extraction")
parser.add_argument('--tuning', default="full", help="choose between <full> network training or <ft> for fine-tuning."
                                                     " If you choosed ResNet or ResneXt as network you can specify the "
                                                     "starting <layer> (layer3, layer4, fc ...) for finetuning")
parser.add_argument('--classify', action="store_true", default=False, help="use choosed network for classification,"
                                                                           " without tripletloss")
parser.add_argument('--margin', default=1., help="triplet loss margin", type=float)
parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
parser.add_argument('--decay', default=1e-4, help="weight decay for Adam optimizer", type=float)
parser.add_argument('--triplets', default="batch-hard", help="choose triplet selector between <batch-hard>,<semi-hard>"
                                                             " and <random-negative>")
parser.add_argument('--epochs', default=20, help="n of training epochs", type=int)
parser.add_argument('--log', default=50, help="log interval length", type=int)
parser.add_argument('--checkpoint', action="store_true", default=False, help="resume training from a checkpoint")
parser.add_argument('--eval', default="gpu", help="choose testing hardware between <gpu>,<cpu> or <vram-opt>")
parser.add_argument('--thresh', default=20, help="discard threshold for ttr,ftr metrics", type=int)
parser.add_argument('--rank', default=20, help="max cmc rank", type=int)
parser.add_argument('--restart', action="store_true", default=False, help="resume evaluation from a pickle dump")

if __name__ == '__main__':
    args = parser.parse_args()

    if args.data_aug == 1:
        transform_list = [transforms.RandomHorizontalFlip(1), transforms.RandomCrop(224, 64)]
        print("Data augmentation...")
        offline_data_aug(transform_list, args.train_max, args.train_min)
    if args.data_aug == 2:
        transform_list = [transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1),
                          transforms.RandomCrop(224, 64)]
        print("Data augmentation...")
        offline_data_aug(transform_list, args.train_max, args.train_min)

    print("Loading train dataset...")
    if args.network == 'inception':
        train_dataset = TVReID(train=True, pid_max=args.train_max, pid_min=args.train_min, is_inception=True)
    else:
        train_dataset = TVReID(train=True, pid_max=args.train_max, pid_min=args.train_min)
    print("Loading test dataset...")
    if args.network == 'inception':
        test_dataset = TVReID(train=False, pid_max=args.test_max, pid_min=args.test_min, non_target=args.non_target,
                              is_inception=True)
    else:
        test_dataset = TVReID(train=False, pid_max=args.test_max, pid_min=args.test_min, non_target=args.non_target)

    print("Loading model...")
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if args.classify:
        train_dataset.classify_train_labels()
        test_dataset.classify_test_labels()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.samples, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.samples, shuffle=False, **kwargs)
    else:
        train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=args.classes, n_samples=args.samples)
        test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=args.classes, n_samples=args.samples)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    if args.network == 'resnet':
        model = EmbeddingResNet(args.tuning)
    if args.network == 'vgg16':
        model = EmbeddingVgg16(args.tuning)
    if args.network == 'inception':
        model = EmbeddingInception(args.tuning)
    if args.network == 'alexnet':
        model = EmbeddingAlexNet(args.tuning)
    if args.network == 'densenet':
        model = EmbeddingDenseNet(args.tuning)
    if args.network == 'resnext':
        model = EmbeddingResNeXt(args.tuning)
    if args.network == 'googlenet':
        model = EmbeddingGoogleNet(args.tuning)
    if args.classify:
        model.classification_setup(args.train_max - 1)  # dataset object samples train_max - 1 pids

    if cuda:
        model.cuda()
    if args.classify:
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        margin = args.margin
        if args.triplets == 'batch-hard':
            loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
        if args.triplets == 'semi-hard':
            loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin))
        if args.triplets == 'random-negative':
            loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = args.epochs
    log_interval = args.log

    if args.checkpoint:
        checkpoint = torch.load('checkpoint.pth.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        print("Restarting training from checkpoint...")
        if args.classify:
            fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, start_epoch=epoch)
        else:
            fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
                metrics=[AverageNonzeroTripletsMetric()], start_epoch=epoch)
    else:
        print("Starting training phase...")
        if args.classify:
            fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
        else:
            fit(train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
                metrics=[AverageNonzeroTripletsMetric()])

    if args.classify:
        print("Starting evaluation phase...")
        classification(train_loader, test_loader, model, cmc_rank=args.rank, n_classes=args.train_max - 1)
    else:
        if args.restart:
            print("Restarting evaluation from dump...")
        else:
            print("Starting evaluation phase...")
        if args.eval == "gpu":
            evaluate_gpu(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank,
                         restart=args.restart)
        if args.eval == "cpu":
            evaluate(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank, restart=args.restart)
        if args.eval == "vram-opt":
            evaluate_vram_opt(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank,
                              restart=args.restart)
