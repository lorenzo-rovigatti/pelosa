'''
Created on 27 set 2019

@author: lorenzo
'''

COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "violet", "brown", "pink", "black", "grey"]            
def print_cogli1_conf(system, colors, filename):
    with open(filename, "w") as output:
        print(".Box:%lf,%lf,%lf" % tuple(system.box), file=output)
        for i, p in enumerate(system.particles()):
            print("%lf %lf %lf @ 0.5 C[%s]" % (p.position[0], p.position[1], p.position[2], colors[i]), file=output)
