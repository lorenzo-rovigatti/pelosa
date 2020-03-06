#!/usr/bin/env python3

'''
Created on 27 set 2019

@author: lorenzo
'''

import warnings
# this might be dangerous
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import sys
import glob

import autoencoder as ae

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage is %s pattern" % sys.argv[0], file=sys.stderr)
        exit(0)

    bops = {}
    for filename in glob.glob(sys.argv[1]):
        timestep = int(filename.split("_")[-1])
        with open(filename) as f:
            bops[timestep] = np.loadtxt(filename)            
        
    all_bops = np.vstack(list(bops.values()))
    
    import tensorflow as tf

    batch_size = 32
    epochs = 2
    learning_rate = 1e-2
    momentum = 9e-1
    code_dim = 2
    original_dim = all_bops.shape[1]
    hidden_dim = all_bops.shape[1] // 2
    hidden_dim = 300
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
            encoded_bops = autoencoder.encoder(tf.constant(bops[timestep]))
            
            np.savetxt(out_file, encoded_bops)
            print("\n", file=out_file)
        
