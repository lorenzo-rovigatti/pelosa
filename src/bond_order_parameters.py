#!/usr/bin/python3

import parsers
import baggianalysis as ba
import sys
import os

def print_dir_bops_to_dir(input_dir, pattern, output_dir, nf=None):
    parser = parsers.Tip4pIceParser()
    # we need a FullTrajectory (as opposed to a LazyTrajectory) because we later use systems as dictionary keys
    trajectory = ba.FullTrajectory(parser)
    trajectory.initialise_from_folder(input_dir, pattern, True)
    
    if nf == None:
        #nf = ba.CutoffFinder(1.3)
        #nf = ba.SANNFinder(3.0, ba.SANNFinder.SYMMETRISE_BY_REMOVING)
        nf = ba.FixedNumberFinder(16)

    syst = trajectory.next_frame()
    while syst != None:
        nf.set_neighbours(syst.particles(), syst.box)
        orders = {2, 4, 6, 8, 10, 12}
        bop_obs = ba.BondOrderParameters(orders, compute_ws=True)
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
    
