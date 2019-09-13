import numpy as np
import torch
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class TVReID(Dataset):

    path_reid = "data/train/"
    path_reid_test = "data/test/"
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    def __init__(self, train, pid_max, pid_min, non_target=0):
        """
        :param train: load train dataset(true) or test dataset(false)
        :param pid_max: if pid_min is omitted all pids from 1 or 0 to pid_max are loaded
        :param pid_min: with pid_max defines the range of pids to load
        :param non_target: n of pids that'll be considered impostors, they'll be sampled from pids after pid_max
        """
        self.train = train
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

        if self.train:
            for i in range(pid_min, pid_max):
                paths = glob.glob(self.path_reid + "Image-{}-*.jpg".format(i))
                if len(paths) > 0:
                    for img in paths:
                        self.train_data.append(img)  # path singola immagine
                        self.train_labels.append(i)  # rispettiva label o id persona
                else:
                    print("ID {} not found!!".format(i))
                if i % 100 == 0:
                    print(i)

            self.train_labels = torch.LongTensor(self.train_labels)
            self.data_len = len(self.train_data)
        else:
            for i in range(pid_min, pid_max):
                paths = glob.glob(self.path_reid_test + "Image-{}-*.jpg".format(i))
                if len(paths) > 0:
                    for img in paths:
                        self.test_data.append(img)  # path singola immagine
                        self.test_labels.append(i)  # rispettiva label o id persona
                else:
                    print("ID {} not found!!".format(i))
                if i % 100 == 0:
                    print(i)
            if non_target > 0:
                for i in range(pid_max, pid_max + non_target):
                    paths = glob.glob(self.path_reid_test + "Image-{}-*.jpg".format(i))
                    if len(paths) > 0:
                        for img in paths:
                            self.test_data.append(img)
                            self.test_labels.append(0)  # label indicante impostore
                    else:
                        print("ID {} not found!!".format(i))
                    if i % 100 == 0:
                        print(i)
            self.test_labels = torch.LongTensor(self.test_labels)
            self.data_len = len(self.test_data)

    def __getitem__(self, index):  # viene caricata l'immagine richiesta dal sampler
        if self.train:
            single_image_name = self.train_data[index]
            img_as_img = Image.open(single_image_name)
            img_as_tensor = self.transform(img_as_img)
            single_image_label = self.train_labels[index]
        else:
            single_image_name = self.test_data[index]
            img_as_img = Image.open(single_image_name)
            img_as_tensor = self.transform(img_as_img)
            single_image_label = self.test_labels[index]

        return img_as_tensor, single_image_label

    def __len__(self):
        return self.data_len


class TripletTVReID(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, tvreid_dataset):
        self.dataset = tvreid_dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        if self.train:
            self.train_labels = self.dataset.train_labels
            self.train_data = self.dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.dataset.test_labels
            self.test_data = self.dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.open(img1)
        img2 = Image.open(img2)
        img3 = Image.open(img3)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


def offline_data_aug(data_aug, pidmax, pidmin=0):
    """
    Gets a list data_aug of PyTorch transforms then applies them to the dataset.
    Result images are added to the original dataset.
    """
    path_reid = "data/train/"

    for i in range(pidmin, pidmax):
        paths = glob.glob(path_reid + "Image-{}-*.jpg".format(i))
        j = len(paths)
        if j > 0:
            for transform in data_aug:
                for img in paths:
                    img_as_img = Image.open(img)
                    img_aug = transform(img_as_img)
                    img_aug.save(path_reid + "Image-{}-{}.jpg".format(i, j))
                    j += 1
        else:
            print("ID {} not found!!".format(i))
        if i % 100 == 0:
            print(i)
