import torch
import pickle
import numpy as np
from metrics import average_precision, cmc_score, open_set_scores


def classification(test_loader, model, cmc_rank, n_classes):
    cuda = torch.cuda.is_available()
    test_lab = []
    test_scores = np.array([], dtype=np.int64).reshape(0, n_classes)
    rank_list = []
    match_list = []
    ap_list = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if cuda:
                images = images.cuda()
            outputs = model(images)
            test_lab = np.append(test_lab, labels.cpu().numpy())
            test_scores = np.concatenate((test_scores, outputs.data.cpu().numpy()), axis=0)

    tot = len(test_lab)
    for i in range(0, tot):
        pred_labels = np.argsort(-test_scores[i])
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):  # pre-ranking of query vector
            if pred_labels[k] == test_lab[i]:
                rank = k + 1

        rank_list.append(rank)
        match_list.append(pred_labels[0])
        ap_list.append(ap)
        print('\r Test {} of {}'.format(i + 1, tot), end="")
    print("")
    print("mAP: {}%".format(np.mean(ap_list) * 100))
    cmc_score(rank_list, rank_max=cmc_rank)


def evaluate(train_dataset, test_dataset, model, thresh, cmc_rank, restart=False):
    """
    :param train_dataset: train dataset object
    :param test_dataset: test dataset object
    :param model: model object
    :param restart: resume evaluation from a pickle dump if True
    :param thresh: discard distance for ttr,ftr metrics
    :param cmc_rank: max cmc rank
    :return:
    """
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs)
    train_emb, train_lab = extract_embeddings(train_loader, model, cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
    test_emb, test_lab = extract_embeddings(test_loader, model, cuda)

    rank_list = []  # contains ranks of every test/query vector
    dist_list = []  # a single element is the shortest distance found between the vector query and all the train vectors
    match_list = []  # contains the predicted labels of the query vectors, rank1 match
    ap_list = []  # contains the average precision of the query vectors
    start = 0
    tot = len(test_lab)
    if restart:
        with open('dump.pkl', 'rb') as f:
            data = pickle.load(f)  # backup loading
            rank_list = data[0]
            dist_list = data[1]
            match_list = data[2]
            ap_list = data[3]
        start = len(rank_list) - 1

    for i in range(start, tot):
        # single query preparation for matrix calculation, more efficient than multiplying all queries
        query_vec = np.reshape(test_emb[i], [1, 1000])
        # distance calculation between query vector e training/gallery feature vector
        dist_vec = -2 * np.dot(query_vec, train_emb.T) + np.sum(train_emb ** 2, axis=1) + np.sum(query_vec ** 2, axis=1)[:, np.newaxis]
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]  # contains the predicted labels ordered according to their distance
        dist_vec = np.sort(dist_vec.flatten())  # distance vector ordered by the smallest

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):  # pre-ranking of query vector
            if pred_labels[k] == test_lab[i]:
                rank = k + 1

        rank_list.append(rank)
        dist_list.append(dist_vec[0])
        match_list.append(pred_labels[0])
        ap_list.append(ap)
        print('\r Test {} of {}'.format(i + 1, tot), end="")
        if (i % 1000) == 0:
            with open('dump.pkl', 'wb') as f:
                data = [rank_list, dist_list, match_list, ap_list]
                pickle.dump(data, f)  # backup saving

    print("")
    print("mAP: {}%".format(np.mean(ap_list)*100))
    cmc_score(rank_list, rank_max=cmc_rank)
    open_set_scores(match_list, dist_list, test_lab, thresh)


def evaluate_vram_opt(train_dataset, test_dataset, model, thresh, cmc_rank, restart=False):
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs)
    train_emb, train_lab = extract_embeddings_gpu(train_loader, model, cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
    test_emb, test_lab = extract_embeddings_gpu(test_loader, model, cuda)

    rank_list = []
    dist_list = []
    match_list = []
    ap_list = []
    start = 0
    tot = len(test_lab)
    if restart:
        with open('dump.pkl', 'rb') as f:
            data = pickle.load(f)
            rank_list = data[0]
            dist_list = data[1]
            match_list = data[2]
            ap_list = data[3]
        start = len(rank_list) - 1

    for i in range(start, tot):
        # single query preparation for matrix calculation, more efficient than multiplying all queries
        query_vec = test_emb[i].view(1, 1000)
        # distance calculation between query vector e training/gallery feature vector
        dist_vec = -2 * torch.mm(query_vec, torch.t(train_emb)) + torch.sum(torch.pow(train_emb, 2), dim=1) + torch.sum(torch.pow(query_vec, 2), dim=1).view(-1,1)
        dist_vec = dist_vec.cpu().numpy()
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]
        dist_vec = np.sort(dist_vec.flatten())

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):  # pre-ranking dell'attuale query vector
            if pred_labels[k] == test_lab[i]:
                rank = k + 1

        rank_list.append(rank)
        dist_list.append(dist_vec[0])
        match_list.append(pred_labels[0])
        ap_list.append(ap)
        print('\r Test {} of {}'.format(i + 1, tot), end="")
        if (i % 1000) == 0:
            with open('dump.pkl', 'wb') as f:
                data = [rank_list, dist_list, match_list, ap_list]
                pickle.dump(data, f)

    print("")
    print("mAP: {}%".format(np.mean(ap_list)*100))
    cmc_score(rank_list, rank_max=cmc_rank)
    open_set_scores(match_list, dist_list, test_lab, thresh)


def evaluate_gpu(train_dataset, test_dataset, model, thresh, cmc_rank, restart=False):  # con 1000 persone non bastano 16gb VRAM
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs)
    train_emb, train_lab = extract_embeddings_gpu(train_loader, model, cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
    test_emb, test_lab = extract_embeddings_gpu(test_loader, model, cuda)
    train_emb = train_emb.half()
    test_emb = test_emb.half()

    dists = -2 * torch.mm(test_emb, torch.t(train_emb)) + torch.sum(torch.pow(train_emb, 2), dim=1) + torch.sum(torch.pow(test_emb, 2), dim=1).view(-1,1)
    dists = dists.cpu().numpy()

    rank_list = []
    dist_list = []
    match_list = []
    ap_list = []
    start = 0
    tot = len(test_lab)

    for i in range(start, tot):
        dist_vec = np.array(dists[i], dtype=float)
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]
        dist_vec = np.sort(dist_vec.flatten())

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):
            if pred_labels[k] == test_lab[i]:
                rank = k + 1

        rank_list.append(rank)
        dist_list.append(dist_vec[0])
        match_list.append(pred_labels[0])
        ap_list.append(ap)
        print('\r Test {} of {}'.format(i + 1, tot), end="")

    print("")
    print("mAP: {}%".format(np.mean(ap_list) * 100))
    cmc_score(rank_list, rank_max=cmc_rank)
    open_set_scores(match_list, dist_list, test_lab, thresh)


def extract_embeddings(dataloader, model, cuda):
    """
    Extracts feature vectors of image and respective labels from the dataloader
    """
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 1000))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def extract_embeddings_gpu(dataloader, model, cuda):
    with torch.no_grad():
        model.eval()
        embeddings = torch.zeros((len(dataloader.dataset), 1000)).cuda()
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k + len(images)] = model.get_embedding(images).data
            labels[k:k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels
