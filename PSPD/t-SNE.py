import random
import numpy as np
import torch
from utils.utils import set_seed, create_model
from utils.get_dataset import get_finetune_dataloader
from utils.config import finetune_config
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn import manifold, metrics, cluster


def get_accuracy_score(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = (np.array(ind)).transpose()
    ac = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    # print('AC = ', ac)
    return ac


def visualize_data(data, labels, title, num_clusters):  # feature visualization
    labels = labels.astype(int)
    tsne = manifold.TSNE(n_components=2)  # init='pca'
    data_tsne = tsne.fit_transform(data)
    fig = plt.figure(figsize=(6.3, 5))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], lw=0, s=10, c=labels, cmap=plt.cm.get_cmap("jet", num_clusters))
    plt.colorbar(ticks=range(num_clusters))
    fig.savefig(title, dpi=600)


def obtain_embedding_feature_map(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        feature_map = []
        target_output = []
        for data, target in test_dataloader:
            data = data.to(device)
            output = model(data).cpu()
            # feature_map[len(feature_map):len(output) - 1] = output.tolist()
            # target_output[len(target_output):len(target) - 1] = target.tolist()
            feature_map.append(output)
            target_output.append(target)
        feature_map = torch.cat(feature_map, dim=0)
        target_output = np.concatenate(target_output, axis=0)
    return feature_map, target_output.astype(np.uint8)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    RANDOM_SEED = 2023  # any random number
    set_seed(RANDOM_SEED)

    png = "t-sne_10dB_13PT_13FT"
    config = finetune_config(shot=100, input_type='cwd', dataset_name="radar_10db")
    config["dataset"]["root"] = "E:/Datasets/Radar/10dB/transformed"
    config["encoder"]["pretrain_path"] = "runs/pretext/ResNet18_radar_10db_cwd_powerNorm/0304_160042/best_encoder.pth"
    config["dataset"]["num_classes"] = 5
    png = "t-sne_10dB_8PT_5FT"

    n_clusters = config["dataset"]["num_classes"]
    model = create_model(config["encoder"]["root"], feature_dim=config["encoder"]["feature_dim"], dtype=config["dataset"]["type"])
    device = config["device"]
    model = model.to(device)

    model.load_state_dict(torch.load(config["encoder"]["pretrain_path"]))

    dataloader, _ = get_finetune_dataloader(config)

    X_test_feature, target = obtain_embedding_feature_map(model, dataloader, device)
    cluster_target = cluster.KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED).fit_predict(X_test_feature)

    visualize_data(X_test_feature, target, png, n_clusters)

    sc = metrics.silhouette_score(X_test_feature, target)
    nmi = metrics.normalized_mutual_info_score(target, cluster_target)
    ari = metrics.adjusted_rand_score(target, cluster_target)
    ac = get_accuracy_score(target, cluster_target)
    print(
        f"Silhouette Coefficient (SC)={sc}\nNormalized Mutual Info (NMI)={nmi}\nAdjusted Rrand Index (ARI)={ari}\nAccuracy Score (AC)={ac}")


if __name__ == "__main__":
    main()
