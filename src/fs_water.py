#!/usr/bin/env python3

'''
Created on 27 set 2019

@author: lorenzo
'''

import warnings
# this might be dangerous
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import sys
import glob
from sklearn.preprocessing import StandardScaler

import autoencoder as ae

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage is %s pattern n_columns normalise [SANN_neigh_dir]" % sys.argv[0], file=sys.stderr)
        exit(0)

    bops = {}
    n_columns = int(sys.argv[2])

    for filename in glob.glob(sys.argv[1]):
        timestep = int(filename.split("_")[-1])
        with open(filename) as f:
            bops[timestep] = np.loadtxt(filename)[:,0:n_columns]
            N_per_conf = bops[timestep].shape[0]
            
    if len(sys.argv) > 4:
        SANN_neighs = []
        SANN_neigh_dir = sys.argv[4]
        base = os.path.join(SANN_neigh_dir, "sann_neighs_")
        for timestep in bops:
            SANN_neighs.extend(np.loadtxt(base + str(timestep)))
    else:
        SANN_neighs = None
        
    all_bops = np.vstack(list(bops.values()))
    normalise = bool(sys.argv[3])
    if normalise:
        scaler = StandardScaler()
        scaler.fit(all_bops)
        all_bops = scaler.transform(all_bops)
    
    import tensorflow as tf

    batch_size = 32
    epochs = 2
    learning_rate = 1e-2
    momentum = 9e-1
    code_dim = 2
    original_dim = all_bops.shape[1]
    hidden_dim = all_bops.shape[1] // 2
    hidden_dim = 100
    weight_lambda = 1e-5
    
    training_dataset = tf.data.Dataset.from_tensor_slices(all_bops).batch(batch_size)
    
    autoencoder = ae.Autoencoder(original_dim=original_dim, code_dim=code_dim, hidden_dim=hidden_dim, weight_lambda=weight_lambda)
    #opt = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    
    with open("loss_tip4p.dat", "w") as loss_file:
        real_step = 0
        for epoch in range(epochs):
            for step, batch_features in enumerate(training_dataset):
                ae.train(ae.loss, autoencoder, opt, batch_features)
                loss_value = ae.loss(autoencoder, batch_features)
                real_step += 1
                print("%d %f" % (real_step, loss_value), file=loss_file)
            print("%d/%d epochs done" % (epoch + 1, epochs), file=sys.stderr)
           
    with open("encoded_tip4p_bops.dat", "w") as out_file:
        for timestep in sorted(bops):
            if normalise:
                original_bops = scaler.transform(bops[timestep])
            else:
                original_bops = bops[timestep]
            encoded_bops = autoencoder.encoder(tf.constant(original_bops))
            try:
                all_encoded_bops = np.concatenate((all_encoded_bops, encoded_bops))
            except:
                all_encoded_bops = np.copy(encoded_bops)
            
            np.savetxt(out_file, encoded_bops)
            print("\n", file=out_file)
            
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=0).fit(all_encoded_bops)
    
    f_0 = 0.
    N_conf = 0
    with open("time_series.dat", "w") as out_file, open("bops_particle_A.dat", "w") as A_file, open("bops_particle_B.dat", "w") as B_file:
        for i, bop in enumerate(all_encoded_bops):
            to_print = " ".join(str(x) for x in bop)
            if SANN_neighs != None:
                to_print += " " + SANN_neighs[i]
            if kmeans.labels_[i] == 0:
                f_0 += 1
                print(to_print, file=A_file)
            else:
                print(to_print, file=B_file)
                
            if i > 0 and i % N_per_conf == 0:
                f_0 /= N_per_conf
                print(N_conf, f_0, file=out_file)
                print("", file=A_file)
                print("", file=B_file)
                
                N_conf += 1
