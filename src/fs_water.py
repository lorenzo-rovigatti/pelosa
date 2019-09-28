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
    def parse(self, conf):
        syst = ba.System()
        
        with open(conf) as f:
            # discard the first line
            f.readline()
            
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
        print("Usage is %s print|analyse input [bops]" % sys.argv[0], file=sys.stderr)
        exit(0)

    input_filename = sys.argv[2]
        
    parser = Tip4pIceParser()
    syst = parser.parse(input_filename)

    if sys.argv[1] == "print":
        #nf = ba.CutoffFinder(1.3)
        nf = ba.SANNFinder(3.0, ba.SANNFinder.SYMMETRISE_BY_REMOVING)
        nf.set_neighbours(syst.particles(), syst.box)
        
        bop_obs = ba.BondOrderParameters({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
        bops = np.array(bop_obs.compute(syst))
        np.savetxt("tip4p_bops.dat", bops)
        
    else:
        bops = np.loadtxt(sys.argv[3])
        
        import tensorflow as tf

        batch_size = 32
        epochs = 10
        learning_rate = 1e-2
        momentum = 9e-1
        code_dim = 2
        original_dim = bops.shape[1]
        hidden_dim = min(original_dim * 10, 100)
        weight_lambda = 1e-5
        
        training_dataset = tf.data.Dataset.from_tensor_slices(bops).batch(batch_size)
        
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
                
        encoded_bops = autoencoder.encoder(tf.constant(bops))
        
        # FIND THE GAUSSIAN MIXTURE MODEL THAT BEST APPROXIMATES THE DATA THROUGH A BIC CRITERIUM
        
        from sklearn import mixture
        import itertools
        
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 10)
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full")
            gmm.fit(encoded_bops)
            bic.append(gmm.bic(encoded_bops))
            # we choose the GMM with the lowest bic
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
        
        np.savetxt("bic_tip4p.dat", list(zip(n_components_range, bic)))
        
        clf = best_gmm
        cluster_probabilities = np.array(clf.predict_proba(encoded_bops))
        
        def entropy(prob):
            my_prob = prob[prob > 0.]
            return -np.sum(my_prob * np.log(my_prob))
        
        def reduce_prob(prob_Y, to_reduce):
            # copy the array
            new_prob_Y = np.array(prob_Y)
            # sum the columns relative to the two clusters
            new_prob_Y[:,to_reduce[0]] += new_prob_Y[:,to_reduce[1]]
            # remove the column relative to the second element of the pair
            return np.delete(new_prob_Y, to_reduce[1], axis=1)
        
        
        Ng = cluster_probabilities.shape[1]
        
        print("The number of initial gaussians is %d" % Ng)
        
        results = {}
        results[Ng] = (np.copy(cluster_probabilities), entropy(cluster_probabilities))
        
        # REDUCE THE NUMBER OF CLUSTERS FROM Ng TO 1
        
        # we reduce the number of gaussians by using the method of Baudry et al (2010)
        while Ng > 1:
            clusters = list(range(Ng))
        
            min_entropy = np.infty
            # loop over all possible cluster pairs
            for to_reduce in itertools.combinations(clusters, 2):
                # sum the probabilities associated to the two clusters to be merged
                new_probabilities = reduce_prob(results[Ng][0], to_reduce)
                # calculate the new entropy
                new_entropy = entropy(new_probabilities)
                # keep track of the pair whose merging yields the smallest entropy
                if new_entropy < min_entropy:
                    min_entropy = new_entropy
                    min_probabilities = np.copy(new_probabilities)
        
            Ng -= 1
            results[Ng] = (min_probabilities, min_entropy)
            
        with open("baudry_tip4p.dat", "w") as baudry_file:
            for Ng in sorted(results.keys()):
                print(Ng, results[Ng][1], file=baudry_file)
                colors = list(map(lambda x: utils.COLORS[np.argmax(x)], results[Ng][0]))
                utils.print_cogli1_conf(syst, colors, "gauss_%d_tip4p.dat" % Ng)
                
                with open("group_%d_tip4p.dat" % Ng, "w") as group_file:
                    colors = np.array(colors)
                    unique_colors = np.unique(colors)
                    for i, color in enumerate(unique_colors):
                        indexes = np.where(colors == color)[0]
                        print("%d - %s" % (i, " ".join([str(x) for x in indexes])), file=group_file)
                    
                
        # DETECT THE "BEST" CLUSTER ACCORDING TO THE L-METHOD OF SALVADOR AND CHAN (2004)
        
        class FitResult(object):
            def __init__(self, slope, intercept, x):
                self.slope = slope
                self.intercept = intercept
                self.y = slope * x + intercept
        
        def best_fit(data):
            x = data[:,0]
            y = data[:,1]
            
            slope, intercept = np.polyfit(x, y, 1)
            
            return FitResult(slope, intercept, x)
        
        def root_mean_squared_error(y1, y2):
            return np.sqrt(np.sum((y1 - y2)**2.) / len(y1))
        
        data = []
        for k in sorted(results):
            data.append([k, results[k][1]])
        data = np.array(data)
        N_partitions = data.shape[0]
        if N_partitions < 4:
            print("There are fewer than 4 clusters (%d), the l-method cannot be applied" % N_partitions, file=sys.stderr)
            exit(1)
        
        position = 2
        min_rmse = np.infty
        while N_partitions - position >= 2:
            left = data[0:position]
            right = data[position:]
        
            left_fit = best_fit(left)
            right_fit = best_fit(right)
        
            left_rmse = root_mean_squared_error(left[:,1], left_fit.y)
            right_rmse = root_mean_squared_error(right[:,1], right_fit.y)
        
            total_rmse = position / N_partitions * left_rmse + (N_partitions - position) / N_partitions * right_rmse
            print("l-method, N_clusters: %d, rmse: %lf" % (data[position,0], total_rmse), file=sys.stderr)
            if total_rmse < min_rmse:
                min_rmse = total_rmse
                min_position = position
                min_crossing = (right_fit.intercept - left_fit.intercept) / (left_fit.slope - right_fit.slope)
            
            position += 1
        
        best_cluster_number = round(min_crossing)
        print("The best number of clusters according to the l-method is %d (crossing position: %lf, rmse: %lf)" % (best_cluster_number, min_crossing, min_rmse), file=sys.stderr)
        
        # here we assume that the number of clusters always starts at 1
        best_result = results[int(best_cluster_number)]
        clusters = list(map(lambda x: np.argmax(x), best_result[0]))
        to_sort = np.c_[clusters, encoded_bops]
        sorted_data = to_sort[to_sort[:,0].argsort()]
        
        last_c = 0
        with open("nn_bops_tip4p.dat", "w") as bops_file:
            for v in sorted_data:
                if v[0] > 0 and v[0] != last_c:
                    print("\n", file=bops_file)
                    last_c = v[0]
                print(" ".join([str(x) for x in v[1:]]), file=bops_file)

