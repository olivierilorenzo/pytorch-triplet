import torch
import pickle
import numpy as np


def evaluate(train_dataset, test_dataset, model, restart=False, thresh=0):
    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, **kwargs)
    train_emb, train_lab = extract_embeddings(train_loader, model, cuda)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)
    test_emb, test_lab = extract_embeddings(test_loader, model, cuda)

    rank_list = []  # contiene i rank di ogni test o query vector
    dist_list = []  # un singolo elemento è la distanza minore trovata tra il query vector e tutti i train vector
    match_list = []  # contiene le label predette dei query vector, rank1 match
    start = 0
    tot = len(test_emb)
    if restart:
        with open('dump.pkl', 'rb') as f:
            rank_list = pickle.load(f)
        start = len(rank_list) - 1

    for i in range(start, tot):
        # preparazione singolo query per il calcolo matriciale, più efficiente che mettere in matrice tutti i query
        query_vec = np.reshape(test_emb[i], [1, 1000])
        # calcolo distanza tra query vector e feature vector di training
        dist_vec = -2 * np.dot(query_vec, train_emb.T) + np.sum(train_emb ** 2, axis=1) + np.sum(query_vec ** 2, axis=1)[:, np.newaxis]
        pred_labels = train_lab[dist_vec.flatten().argsort()]
        pred_labels = pred_labels[:20]  # contiene le label predette ordinate secondo la rispettiva distanza
        dist_vec = np.sort(dist_vec.flatten())  # vettore distanze ordinato dalla piu piccola

        rank = 0
        for k in range(len(pred_labels) - 1, -1, -1):  # pre-ranking dell'attuale query vector
            if pred_labels[k] == test_lab[i]:
                rank = k + 1

        rank_list.append(rank)
        dist_list.append(dist_vec[0])
        match_list.append(pred_labels[0])
        print('\r Test {} of {}'.format(i + 1, tot), end="")
        if (i % 1000) == 0:
            with open('dump.pkl', 'wb') as f:
                pickle.dump(rank_list, f)  # salvataggio della rank list su memoria

    num_rank = 0
    print("")
    for j in range(1, len(pred_labels)+1):  # calcolo CMC
        num_rank += rank_list.count(j)
        rank_value = (num_rank / len(rank_list)) * 100
        print("Rank {}: {}%".format(j, rank_value))

    non_target_tot = list(test_lab).count(0)
    target_tot = tot - non_target_tot
    target = 0
    non_target = 0
    for z in range(tot):  # calcolo TTR e FTR
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

    return None


def extract_embeddings(dataloader, model, cuda):
    """
    Estrae i feature vector delle immagini e le rispettive label
    dal dataloader utilizzato.
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
