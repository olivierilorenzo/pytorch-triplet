# Person Re-ID with Triplet Loss

PyTorch implementation of Triplet Loss using a pre-trained ResNet50 as baseline network. 

Base code of Triplet Loss and Triplet Mining is taken from adambielski's repository:

https://github.com/adambielski/siamese-triplet

## Required packages
- python 3.5
- pytorch 1.1.0 or later
- torchvision 0.3.0 or later
- pillow 6.0.0 or later
- numpy 1.16.3 or later

## Getting started

### File directory

```
  pytorch-triplet
     ├── data                    
         ├── train          
         └── test 
     ├── main.py
     ├── datasets.py
     ├── evaluation.py
     ├── losses.py
     ├── metrics.py
     ├── networks.py
     ├── trainer.py
     └── utils.py
```

You must put your dataset in the data folder split in train and test.

### Script params

- **train_min**, first pid of the training interval, default value 0

- **train_max**, last pid of the training interval

- **test_min**, first pid of the testing interval, default value 0

- **test_max**, last pid of the testing interval

- **data_aug**, applies data augmentation on the train min-max interval,
               <0> no data augmentation, <1> horizontal flip + random crop, <2> horizontal flip + vertical flip + random crop,
               default value 0

- **non_target**, n of impostors to be sampled from dataset, default value 0 

- **classes**, n of classes in the mini-batch, default value 5

- **samples**, n of sample per class in the mini-batch, default value 20

- **tuning**, choose between <full> network training or fine-tuning from a specific ResNet <layer> (layer3, layer4, fc ...)

- **margin**, triplet loss margin, default value 1

- **lr**, learning rate of the network, default value 1e-3

- **decay**, weight decay for Adam optimizer, default value 1e-4

- **triplets**, choose triplet selector between <batch-hard>,<semi-hard> and <random-negative>, default value <batch-hard>

- **epochs**, n of training epochs, default value 20

- **log**, log interval length, default value 50

- **checkpoint**, resume training from a checkpoint

- **eval**, choose testing hardware between <gpu>,<cpu> or <vram-opt>, default value <gpu>

- **thresh**, discard threshold for TTR,FTR metrics, default value 20

- **rank**, max cmc rank, default value 20

- **restart**, resume evaluation from a pickle dump

### Examples

Training from scratch:

```
!python3 main.py --train_min 700 --train_max 800 --test_min 700 --test_max 800 --data_aug 0 --epochs 20 --tuning layer3 --classes 5 --samples 20 --lr 10e-4 --decay 10e-5 
```

Restarting training from a tar checkpoint:

```
!python3 main.py --train_min 700 --train_max 800 --test_min 700 --test_max 800 --epochs 20 --tuning layer3 --classes 5 --samples 20 --lr 10e-4 --decay 10e-5 --checkpoint
```

Restarting network test from a pickle dump:

```
!python3 main.py --train_min 700 --train_max 800 --test_min 700 --test_max 800 --epochs 20 --tuning layer3 --classes 5 --samples 20 --lr 10e-4 --decay 10e-5 --restart
```

If you want to directly test the network simply put the same number of epochs of your checkpoint as epochs param.

### Warning

Be sure to adjust *classes* and *samples* params according to your video card memory. Note that training ResNet50 from scratch will take more memory than finetuning.

Also, pay attention to use *data_aug* param only the first time you launch the script otherwise you'll apply data aumentation on the already augmented dataset .
