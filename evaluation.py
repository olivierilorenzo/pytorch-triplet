import torch
import pickle
import numpy as np


def evaluate(train_dataset, test_dataset, model, thresh, cmc_rank, restart=False):
    """
    :param restart: la valutazione riparte dal dump pickle
    :param thresh: distanza oltre la quale le query vengono considerati impostori e scartati
                  (solo se sono presenti impostori nel dataset)
    :param cmc_rank: rank massimo delle cumulative matching curve
    :return:
    """
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs)
    train_emb, train_lab = extract_embeddings(train_loader, model, cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
    test_emb, test_lab = extract_embeddings(test_loader, model, cuda)

    rank_list = []  # contiene i rank di ogni test o query vector
    dist_list = []  # un singolo elemento è la distanza minore trovata tra il query vector e tutti i train vector
    match_list = []  # contiene le label predette dei query vector, rank1 match
    ap_list = []  # contiene le average precision dei query vector
    start = 0
    tot = len(test_lab)
    if restart:
        with open('dump.pkl', 'rb') as f:
            data = pickle.load(f)  # caricamento backup da file
            rank_list = data[0]
            dist_list = data[1]
            match_list = data[2]
            ap_list = data[3]
        start = len(rank_list) - 1

    for i in range(start, tot):
        # preparazione singolo query per il calcolo matriciale, più efficiente che mettere in matrice tutti i query
        query_vec = np.reshape(test_emb[i], [1, 1000])
        # calcolo distanza tra query vector e feature vector di training
        dist_vec = -2 * np.dot(query_vec, train_emb.T) + np.sum(train_emb ** 2, axis=1) + np.sum(query_vec ** 2, axis=1)[:, np.newaxis]
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]  # contiene le label predette ordinate secondo la rispettiva distanza
        dist_vec = np.sort(dist_vec.flatten())  # vettore distanze ordinato dalla piu piccola

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
                pickle.dump(data, f)  # salvataggio backup su file

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

    rank_list = []  # contiene i rank di ogni test o query vector
    dist_list = []  # un singolo elemento è la distanza minore trovata tra il query vector e tutti i train vector
    match_list = []  # contiene le label predette dei query vector, rank1 match
    ap_list = []  # contiene le average precision dei query vector
    start = 0
    tot = len(test_lab)
    if restart:
        with open('dump.pkl', 'rb') as f:
            data = pickle.load(f)  # caricamento backup da file
            rank_list = data[0]
            dist_list = data[1]
            match_list = data[2]
            ap_list = data[3]
        start = len(rank_list) - 1

    for i in range(start, tot):
        # preparazione singolo query per il calcolo matriciale, più efficiente che mettere in matrice tutti i query
        query_vec = test_emb[i].view(1, 1000)
        # calcolo distanza tra query vector e feature vector di training
        dist_vec = -2 * torch.mm(query_vec, torch.t(train_emb)) + torch.sum(torch.pow(train_emb, 2), dim=1) + torch.sum(torch.pow(query_vec, 2), dim=1).view(-1,1)
        dist_vec = dist_vec.cpu().numpy()
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]  # contiene le label predette ordinate secondo la rispettiva distanza
        dist_vec = np.sort(dist_vec.flatten())  # vettore distanze ordinato dalla piu piccola

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
                pickle.dump(data, f)  # salvataggio backup su file

    print("")
    print("mAP: {}%".format(np.mean(ap_list)*100))
    cmc_score(rank_list, rank_max=cmc_rank)
    open_set_scores(match_list, dist_list, test_lab, thresh)


def evaluate_gpu(train_dataset, test_dataset, model, thresh, cmc_rank, restart=False): # con 1000 persone non bastano 16gb VRAM
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

    rank_list = []  # contiene i rank di ogni test o query vector
    dist_list = []  # un singolo elemento è la distanza minore trovata tra il query vector e tutti i train vector
    match_list = []  # contiene le label predette dei query vector, rank1 match
    ap_list = []  # contiene le average precision dei query vector
    start = 0
    tot = len(test_lab)

    for i in range(start, tot):
        dist_vec = np.array(dists[i], dtype=float)
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        ap = average_precision(test_lab[i], pred_labels)
        pred_labels = pred_labels[:cmc_rank]  # contiene le label predette ordinate secondo la rispettiva distanza
        dist_vec = np.sort(dist_vec.flatten())  # vettore distanze ordinato dalla piu piccola

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):  # pre-ranking dell'attuale query vector
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
    Estrae i feature vector delle immagini e le rispettive label
    dal dataloader utilizzato utilizzando il modello
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


def average_precision(query, pred):
    match = 0
    temp = 0
    for i in range(len(pred)):
        if query == pred[i]:
            match += 1
            temp += match / (i+1)

    if match == 0:
        ap = 0
    else:
        ap = temp/match

    return ap


def cmc_score(rank_list, rank_max):
    num_rank = 0
    print("")
    for j in range(1, rank_max + 1):
        num_rank += rank_list.count(j)
        rank_value = (num_rank / len(rank_list)) * 100
        print("Rank {}: {}%".format(j, rank_value))


def open_set_scores(match_list, dist_list, test_lab, thresh=20): # calcolo TTR e FTR
    tot = len(test_lab)
    non_target_tot = list(test_lab).count(0)
    target_tot = tot - non_target_tot
    target = 0
    non_target = 0
    if non_target_tot > 0:
        for z in range(tot):
            if dist_list[z] < thresh:
                if match_list[z] == test_lab[z]:
                    target += 1
                else:
                    if test_lab[z] == 0:
                        non_target += 1

        ttr = (target / target_tot) * 100
        ftr = (non_target / non_target_tot) * 100
        print("True target rate: {}%".format(ttr))
        print("False target rate: {}%".format(ftr))
