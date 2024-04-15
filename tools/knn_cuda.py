#from sklearn.cluster import KMeans
#  pip install  kmeans_pytorch  -i https://pypi.tuna.tsinghua.edu.cn/simple
from fast_pytorch_kmeans import KMeans
import os
from pathlib import Path
import torch
from einops import rearrange
import numpy as np
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--fea_root_path', help='feature root path', default='/nfs/projects/few_shot/fewshot_fea/xincheng')
    parser.add_argument('--out_path', default='/nfs/projects/few_shot/fewshot_fea/xincheng', help='Path to output file')
    parser.add_argument('--n_clusters', type=int, default=100, help='cluster center num')
    parser.add_argument('--depart', nargs='+', default=['big_18', 'big_shenyang', 'big_penma'], help='depart list')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    fea_root_path = args.fea_root_path
    fea_root_path = Path(fea_root_path)
    out_path = Path(args.out_path)
    depart_names = args.depart
    device = torch.device('cuda:0')

    for depart_name in depart_names:
        save_dict = {}
        fea_path = fea_root_path / '{}.pth'.format(depart_name)
        out_fea_path = out_path / '{}_cluster.pth'.format(depart_name)
        fea = torch.load(fea_path)
        fea1 = fea['feats1']
        fea2 = fea['feats2']
        fea3 = fea['feats3']
        b, c, h1, w1 = fea1.shape
        b, c, h2, w2 = fea2.shape
        b, c, h3, w3 = fea3.shape
        fea1 = rearrange(fea1, 'b c h w -> (b c) (h w)')
        cluster = KMeans(n_clusters=args.n_clusters, mode='euclidean', verbose=1, minibatch=4096)
        # kmeans.fit_predict(feat.to(device))
        cluster.fit(fea1.to(device))
        fea1 = cluster.centroids
        fea1 = fea1.reshape(-1, h1, w1).cpu()
        fea2 = rearrange(fea2, 'b c h w -> (b c) (h w)')
        cluster = KMeans(n_clusters=args.n_clusters, mode='euclidean', verbose=1, minibatch=4096)
        cluster.fit(fea2.to(device))
        fea2 = cluster.centroids
        fea2 = fea2.reshape(-1, h2, w2).cpu()
        fea3 = rearrange(fea3, 'b c h w -> (b c) (h w)')
        cluster = KMeans(n_clusters=args.n_clusters, mode='euclidean', verbose=1, minibatch=4096)
        cluster.fit(fea3.to(device))
        fea3 = cluster.centroids
        fea3 = fea3.reshape(-1, h3, w3).cpu()
        save_dict['feats1'] = fea1
        save_dict['feats2'] = fea2
        save_dict['feats3'] = fea3
        torch.save(save_dict, str(out_fea_path))
    pass
