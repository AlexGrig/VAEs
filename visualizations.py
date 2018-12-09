#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:49:37 2018

@author: alex
"""
import time

def plot_tsne(z_loc, classes, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    print("T-SNE fit started...")
    tt = time.time()
    z_embed = model_tsne.fit_transform(z_loc)
    print("T_SNE fit time {}".format( time.time() - tt) )
    colors = [ 'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'purple', 'olive', 'lime', 'maroon']

    fig = plt.figure()
    for ic in range(10):
        ind_vec = np.zeros_like(classes)
        ind_vec[:, ic] = 1
        ind_class = classes[:, ic] == 1
        #color = plt.cm.Set1(ic)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=colors[ic])
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig('./vae_my_results/'+str(name)+'_embedding_'+str(ic)+'.png')
    fig.savefig('./vae_my_results/'+str(name)+'_embedding.png')
    plt.show()