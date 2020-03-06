'''
Created on 6 mar 2020

@author: lorenzo
'''

import baggianalysis as ba

class Tip4pIceParser(ba.BaseParser):
    def __init__(self):
        ba.BaseParser.__init__(self)
    
    def _parse_file(self, conf):
        syst = ba.System()
        
        with open(conf) as f:
            # we use the first line to check whether we reached the EOF
            first_line = f.readline()
            
            if len(first_line) == 0:
                return None
            
            syst.time = int(conf.split("t")[1])
            
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
    