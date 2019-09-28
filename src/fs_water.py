#!/usr/bin/env python3

'''
Created on 27 set 2019

@author: lorenzo
'''

import baggianalysis as ba
import numpy as np
import sys

import utils
import autoencoder as ae

class Tip4pIceParser(ba.BaseParser):
    def open_parse_close(self, conf):
        syst = ba.System()
        
        with open(conf) as f:
            # we use the first line to check whether we reached the EOF
            if f.readline() == 0:
                return None
            
            N_atoms = int(f.readline().strip())
            for _ in range(N_atoms):
                line = f.readline()
                spl = [x.strip() for x in line.split()]
                if spl[1] == "OW1":
                    pos = [float(x) for x in spl[3:6]]
                    particle = ba.Particle("0", pos, [0., 0., 0.])
                    syst.add_particle(particle)
                    
            syst.box = [float(x.strip()) for x in f.readline().split()]
            
        return syst

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage is %s input_folder [bops]" % sys.argv[0], file=sys.stderr)
        exit(0)

    parser = Tip4pIceParser()
    trajectory = ba.FullTrajectory(parser)
    trajectory.initialise_from_folder(sys.argv[1], "*")

    #if len(sys.argv) == 2:
    syst = trajectory.next_frame()
    while syst != None:
        #nf = ba.CutoffFinder(1.3)
        nf = ba.SANNFinder(3.0, ba.SANNFinder.SYMMETRISE_BY_REMOVING)
        nf.set_neighbours(syst.particles(), syst.box)
        
        bop_obs = ba.BondOrderParameters({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
        syst.bops = np.array(bop_obs.compute(syst))
        
        try:
            all_bops = np.concatenate(all_bops, syst.bops)
        except:
            all_bops = np.copy(syst.bops)
        
        syst = trajectory.next_frame()
        
    #else:
    #    all_bops = np.loadtxt(sys.argv[2])
        
    import tensorflow as tf

    batch_size = 32
    epochs = 10
    learning_rate = 1e-2
    momentum = 9e-1
    code_dim = 2
    original_dim = all_bops.shape[1]
    hidden_dim = min(original_dim * 10, 100)
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
           
    trajectory.reset()
    syst = trajectory.next_frame()
    while syst != None:
        syst.encoded_bops = autoencoder.encoder(tf.constant(syst.bops))
        
        np.savetxt("encoded_tip4p_bops.dat", syst.encoded_bops)
        
        syst = trajectory.next_frame()
        