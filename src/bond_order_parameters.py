#!/usr/bin/python3

import parsers
import baggianalysis as ba
import sys
import os

def print_dir_bops_to_dir(input_dir, pattern, output_dir):
    parser = parsers.Tip4pIceParser()
    # we need a FullTrajectory (as opposed to a LazyTrajectory) because we later use systems as dictionary keys
    trajectory = ba.FullTrajectory(parser)
    trajectory.initialise_from_folder(input_dir, pattern, True)
    
    syst = trajectory.next_frame()
    while syst != None:
        #nf = ba.CutoffFinder(1.3)
        #nf = ba.SANNFinder(3.0, ba.SANNFinder.SYMMETRISE_BY_REMOVING)
        nf = ba.FixedNumberFinder(12)
        nf.set_neighbours(syst.particles(), syst.box)
        
        orders = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
        bop_obs = ba.BondOrderParameters(orders)
        bop_obs.analyse_system(syst)
        bops = bop_obs.result()
        
        out_filename = os.path.join(output_dir, "bops_%d" % syst.time)
        with open(out_filename, "w") as out_file:
            print("# t=%d, orders=%s" % (syst.time, ",".join(str(o) for o in orders)), file=out_file)
            for p_bops in bops:
                print(" ".join([str(x) for x in p_bops]), file=out_file)
        
        syst = trajectory.next_frame()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage is %s input_dir pattern" % sys.argv[0])
        exit(1)
        
    print_dir_bops_to_dir(sys.argv[1], sys.argv[2], ".")
    