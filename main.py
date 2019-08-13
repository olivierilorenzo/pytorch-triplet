import torch
import argparse
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from datasets import offline_data_aug
from datasets import TVReID
from datasets import BalancedBatchSampler
from networks import EmbeddingResNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric
from trainer import fit
from evaluation import evaluate, evaluate_gpu, evaluate_vram_opt

parser = argparse.ArgumentParser()
parser.add_argument('--train_min', help="minimum interval train pid", type=int)
parser.add_argument('--train_max', help="maximum interval  train pid", type=int)
parser.add_argument('--test_min', help="minimum interval test pid", type=int)
parser.add_argument('--test_max', help="maximum interval test pid", type=int)
parser.add_argument('--data_aug', default=0, help="data augmentation", type=int)
parser.add_argument('--non_target', default=0, help="n of impostors", type=int)
parser.add_argument('--classes', default=5, help="n of classes in the mini-batch", type=int)
parser.add_argument('--samples', default=20, help="n of sample per class in the mini-batch", type=int)
parser.add_argument('--mode', default="full", help="choose between -full- network training or "
                                                   "finetuning from a specific resnet -layer-")
parser.add_argument('--margin', default=1., help="triplet loss margin", type=float)
parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
parser.add_argument('--decay', default=1e-4, help="weight decay", type=float)
parser.add_argument('--triplets', default="batch-hard", help="triplet selector, choose between -batch-hard-,-semi-hard-"
                                                             " and -random-negative-")
parser.add_argument('--epochs', default=20, help="n of training epochs", type=int)
parser.add_argument('--log', default=50, help="log interval length", type=int)
parser.add_argument('--checkpoint', action="store_true", default=False, help="resume training from a checkpoint")
parser.add_argument('--eval', default="gpu", help="choose testing hardware between -gpu-,-cpu- or -vram_opt-")
parser.add_argument('--thresh', default=20, help="discard threshold for ttr,ftr metrics", type=int)
parser.add_argument('--rank', default=20, help="cmc rank", type=int)
parser.add_argument('--restart', action="store_true", default=False, help="resume evaluation from a pickle dump")

if __name__ == '__main__':
    args = parser.parse_args()
    # data augmentation
    if args.data_aug == 1:
        transform_list = [transforms.RandomHorizontalFlip(1), transforms.RandomCrop(224, 64)]
        print("Data augmentation...")
        offline_data_aug(transform_list, args.train_max, args.train_min)
    if args.data_aug == 2:
        transform_list = [transforms.RandomHorizontalFlip(1), transforms.RandomVerticalFlip(1),
                          transforms.RandomCrop(224, 64)]
        print("Data augmentation...")
        offline_data_aug(transform_list, args.train_max, args.train_min)

    # preparazione dataset
    print("Loading train dataset")
    train_dataset = TVReID(train=True, pid_max=args.train_max, pid_min=args.train_min)
    print("Loading test dataset")
    test_dataset = TVReID(train=False, pid_max=args.test_max, pid_min=args.test_min, non_target=args.non_target)

    # preparazione modello
    cuda = torch.cuda.is_available()

    train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=args.classes, n_samples=args.samples)
    test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=args.classes, n_samples=args.samples)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    margin = args.margin

    model = EmbeddingResNet(args.mode)

    if cuda:
        model.cuda()
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
        fit(online_train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
            metrics=[AverageNonzeroTripletsMetric()], start_epoch=epoch)
    else:
        print("Starting training phase...")
        fit(online_train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
            metrics=[AverageNonzeroTripletsMetric()])

    if args.restart:
        print("Restarting evaluation from dump...")
    else:
        print("Starting evaluation phase...")
    if args.eval == "gpu":
        evaluate_gpu(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank, restart=args.restart)
    if args.eval == "cpu":
        evaluate(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank, restart=args.restart)
    if args.eval == "vram-opt":
        evaluate_vram_opt(train_dataset, test_dataset, model, thresh=args.thresh, cmc_rank=args.rank, restart=args.restart)
