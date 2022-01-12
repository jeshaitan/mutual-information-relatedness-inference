import sys
import numpy as np
import matplotlib.pyplot as plt

''' 
 usage: python ibd_props.py <output.seg file> <refined simmap file>
'''
seg_filename = sys.argv[1]
#simmap_filename = sys.argv[2]

'''
 read files in as matrices
'''
seg = open(seg_filename)
seg_lines = seg.readlines()
seg_matrix = np.array([np.array(segline.split()) for segline in seg_lines])
print("Finished reading segments.")
#simmap = open(simmap_filename)
#simmap_lines = simmap.readlines()[1:]
#simmap_matrix = np.array([np.array(simmapline.split()) for simmapline in simmap_lines])
#print("Finished reading simmap.")

'''
 splits seg lines into matrices for each pair
'''
def split_pairs(segm):
    groups = []
    current_pair_group = []
    # i.e. ('grandparent1_g1-b1-s1', 'grandparent1_g3-b1-i1')
    current_pair = (segm[0][0], segm[0][1])
    for line in segm:
        if (line[0], line[1]) == current_pair:
            end_on_match = True
        else:
            # new pair encountered, restart accumulator
            if current_pair_group:
                groups.append(current_pair_group)
            current_pair = (line[0], line[1])
            current_pair_group = []
            end_on_match = False
        current_pair_group.append(line)
        
    if(end_on_match):
        groups.append(current_pair_group)
    
    print(len(groups))
    return groups

'''
 removes any relationships including a spouse [-s]
 removes all but one correct pair from each pedigree

def clean_pairs(pairs):
    cleaned_pairs = []
    
    # pedigrees that have already contributed a pair
    pedigrees = []
    for pair in pairs:
        # if either sample in the pair is spousal
        if ('-s' in pair[0][0]) or ('-s' in pair[0][1]):
            continue
        
        # if there is already a pair from this pedigree
        # e.g. pair with 'grandparent1_g3-b1-i1 should not be considered if
        # pair with grandparent1_g3-b2-i1 is already in the set'
        elif pair[0][0].split('_')[0] in pedigrees:
            continue
        else:
            pedigrees.append(pair[0][0].split('_')[0])
            cleaned_pairs.append(pair)

    return cleaned_pairs
'''

# gs = clean_pairs(split_pairs(seg_matrix))
gs = split_pairs(seg_matrix)

'''
 calculates total genetic length given a simmap
'''    
def gen_length(sm):
    acc = 0
    positions = len(sm)
    for l in range(positions):
        # find beginning of new chromosomes
        # previous line (from l = 0) is length of last chromosome
        if sm[l][2] == '0' or l == 0:
            # average the lengths of male and female chr length
            acc += 0.5 * (float(sm[l - 1][2]) + float(sm[l - 1][3]))
            
    return acc

#simmap_length = gen_length(simmap_matrix)

'''
 number of ibd segments is rows in pair
'''
def ibdcount(pair):
    if('fs' in pair[0][0]):
        # dont double count IBD2 segments inside of IBD1
        sandwiched = 0
        for seg_ind in range(2, len(pair)):
            cur = pair[seg_ind]
            last = pair[seg_ind - 1]
            lastlast = pair[seg_ind - 2]
            if(cur[5] == 'IBD1' and last[5] == 'IBD2' and lastlast[5] == 'IBD1'):
                # check for adjacent physical position markers
                if(int(cur[3]) == int(last[4]) + 1 and 
                   int(last[3]) == int(lastlast[4]) + 1):
                    sandwiched += 1
        
        return len(pair) - sandwiched
    else:
        return len(pair)

'''
 proportion of ibd is length shared / total length
'''
def propibd(pair, maplen):
    ibd_length = 0
    for ibd in pair:
        if ibd[5] == 'IBD1':
            ibd_length += 0.5 * float(ibd[8])
        elif ibd[5] == 'IBD2':
            ibd_length += float(ibd[8])
            
    # hard-coded in length from maplen.awk on hapmap bim file
    return ibd_length / maplen
#    return ibd_length / simmap_length 

'''
 label degree of relatedness
'''
def label_relatedness(pair):
    ind1name = pair[0][0]
    if (pair[0][0].split('_')[0] != pair[0][1].split('_')[0]):
        return -1
    elif ('parent-child' in ind1name or 'fs' in ind1name):
        return 1
    elif ('half-sibs' in ind1name or ('AV' in ind1name and (not ('H' in ind1name)))):
        return 2
    elif ('1st-cousins' in ind1name or 'HAV' in ind1name):
        return 3
    elif ('fc1r' in ind1name or 'half-first-cousins' in ind1name):
        return 4
    elif ('2nd-cousins' in ind1name or 'HFC1R' in ind1name):
        return 5
    elif ('sc1r' in ind1name or 'half-second-cousins' in ind1name):
        return 6
    elif ('3C' in ind1name or 'HSC1R' in ind1name):
        return 7 
    else:
        print("Pair not matched: {}".format(ind1name))
        return -1

def name(pair):
    return "{}&{}".format(pair[0][0], pair[0][1])
    
point_pairs = [(propibd(g, 3069.9270), ibdcount(g), label_relatedness(g), name(g)) for g in gs]

import csv
with open(sys.argv[1].split('.')[0] + "nonsel.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    for pair in point_pairs:
        writer.writerow(pair)

# plt.scatter(*zip(*point_pairs))
# plt.xlabel('Proportion IBD')
# plt.ylabel('Count of IBD segments')
# plt.savefig('count_by_prop_ibd.png')
# plt.show()