# Now that we have implement `PCA' from scratch, let's try with more dimensionality reduction methods. PCA is a linear dimensionality reduction method, 
# but real world data includes more non-linear information. `TSNE' and `UMAP' are the two most widely used algorithms for nonlinear dimensionality reduction.

# TODO: Implement `TSNE' and `UMAP' algorithms. It is not easy to implement those two algorithms from scratch, but python packages `sklearn' and `umap-learn' have already implemented
# `TSNE' and `UMAP' for us, use the functions in those packages and fill in the blank part of `tsne_op' and `umap_op'. Note that your implementation of `tsne_op' should be adjustable
# with regarding to hyper-parameter`perplexity', and your implementation of `umap_op` should also be adjustaboe with regarding to  `n_neighbors' and `min_dist'.
# Play around with hyper-parameters in TSNE(`perplexity') and UMAP(`n_neighbors', `min_dist') on counts_PBMC.csv data. Compare the results generated with different hyper-parameters, 
# briefly discuss how those hyper-parameters affect the final visualization results and find the set of hyper-parameters that can generate the best visualization

# Note: 
# 1. To run the code, you need to install one additional package: [umap-learn](https://umap-learn.readthedocs.io/en/latest/)
# 2. Different from hw2_1.py, we didn't include the normalization step for you. Don't forget to normalize the data before doing dimensionality reduction
# 3. ``TSNE'' and ``UMAP'' is computationally expensive, people usually reduce the dimensions of the data to around $30-100$ 
#    using ``PCA'' before they feed the data into ``TSNE'' and ``UMAP''. Make sure you do that too with your implementation.
# 4. plotting function is already provided as `plot_latent'. 

# Additional references:
# 1. sklearn TSNE API: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# 2. sklearn Pipeline API: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
# 3. sklearn PCA API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# 4. sklearn StandardScaler API: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
# 5. Basic usage of umap-learn: https://umap-learn.readthedocs.io/en/latest/basic_usage.html


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from umap import UMAP 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def pca_sklearn(X, n_pcs = 100):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_pcs)
    pca.fit(X)
    return np.dot(X,pca.components_.T) , pca.explained_variance_ratio_

def tsne_op(X, perplexity):
    # implement your code here:
    return TSNE(n_components=2, perplexity=perplexity).fit_transform(X)

def umap_op(X, n_neighbors, min_dist):
    # implement your code here:
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(X)

# plotting function, no modification required
def plot_latent(z, anno, save = None, figsize = (10,10), axis_label = "Latent", **kwargs):
    _kwargs = {
        "s": 10,
        "alpha": 0.9,
    }
    _kwargs.update(kwargs)

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot()
    cluster_types = set([x for x in np.unique(anno)])
    colormap = plt.cm.get_cmap("tab20", len(cluster_types))

    for i, cluster_type in enumerate(cluster_types):
        index = np.where(anno == cluster_type)[0]
        ax.scatter(z[index,0], z[index,1], color = colormap(i), label = cluster_type, **_kwargs)
    
    ax.legend(loc='upper left', prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1))
    
    ax.tick_params(axis = "both", which = "major", labelsize = 15)

    ax.set_xlabel(axis_label + " 1", fontsize = 19)
    ax.set_ylabel(axis_label + " 2", fontsize = 19)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)  
    
    if save:
        fig.savefig(save, bbox_inches = "tight")
    
    print(save)


# read data matrix
expr_ctrl = pd.read_csv("counts_PBMC.csv", sep = ",", index_col = 0).values
anno_ctrl = pd.read_csv("celltypes_PBMC.txt", sep = "\t", header = None)

X_pca_skl, weight = pca_sklearn(expr_ctrl)

for n_neighbors in [5, 15, 30 ,50 ,100]:
 for min_dist in [0, 0.01, 0.05, 0.1, 0.5, 1]:
  x_umap = umap_op(X_pca_skl, n_neighbors = n_neighbors, min_dist = min_dist)
  plot_latent(x_umap, anno_ctrl, axis_label = "UMAP_neighbors_"+str(n_neighbors)+"_min_dist_"+str(min_dist)+"_", save = "part2/UMAP_neighbors_"+str(n_neighbors)+"_min_dist_"+str(min_dist)+".pdf")

for perplexity in [5, 15, 30 ,50 ,100]:
  x_tsne = tsne_op(X_pca_skl, perplexity = perplexity)
  plot_latent(x_tsne, anno_ctrl, axis_label = "TSNE_perplexity_"+str(perplexity)+"_", save = "part2/TSNE_perplexity_"+str(perplexity)+".pdf")


