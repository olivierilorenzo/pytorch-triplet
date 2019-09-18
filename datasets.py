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

    def __init__(self, train, pid_max, pid_min, non_target=0, is_inception=False):
        """
        :param train: load train dataset(true) or test dataset(false)
        :param pid_max: if pid_min is omitted all pids from 1 or 0 to pid_max are loaded
        :param pid_min: with pid_max defines the range of pids to load
        :param non_target: n of pids that'll be considered impostors, they'll be sampled from pids after pid_max
        """
        self.train = train
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if is_inception:
            self.transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        if self.train:
            for i in range(pid_min, pid_max):
                paths = glob.glob(self.path_reid + "Image-{}-*.jpg".format(i))
                if len(paths) > 0:
                    for img in paths:
                        self.train_data.append(img)  # single image path
                        self.train_labels.append(i)  # label/person id
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
                        self.test_data.append(img)  # single image path
                        self.test_labels.append(i)  # label/person id
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
                            self.test_labels.append(0)  # impostor label
                    else:
                        print("ID {} not found!!".format(i))
                    if i % 100 == 0:
                        print(i)
            self.test_labels = torch.LongTensor(self.test_labels)
            self.data_len = len(self.test_data)

    def __getitem__(self, index):  # activated by sampler
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

    def classify_train_labels(self):  # cross-entropy loss takes ids from 0 to Class-1
        labels = self.train_labels.numpy()
        labels = labels - 1
        self.train_labels = torch.LongTensor(labels)

    def classify_test_labels(self):
        labels = self.test_labels.numpy()
        labels = labels - 1
        self.test_labels = torch.LongTensor(labels)


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
