import numpy as np
import matplotlib.pyplot as plt
import sys
import glob, os

#usage: python3 char_trn.py <training csvs/> <features: "p", "n", "pn">
os.chdir(sys.argv[1])
features = sys.argv[2]

#targettrain = 210000
targettrain = 21000

# pairs should be sorted by file by degree
trnpairs = []
trn_inds = [0]
for trnfile in glob.glob("*.csv"):
    trnpairs_i = np.loadtxt(trnfile, delimiter=',', usecols=(0, 1, 2))
    name = trnpairs_i[0][2]
    
    trnsize_i = len(trnpairs_i)
    for pair in trnpairs_i[0 : trnsize_i]:
        if(features == "pn"):
            trnpairs.append(pair)
        elif(features == "p"):
            trnpairs.append(np.delete(pair, 1))
        elif(features == "n"):
            trnpairs.append(np.delete(pair, 0))

    trn_inds.append(trn_inds[-1] + trnsize_i)

#assert len(trnpairs) == targettrain

# normalize all pairs according to training
maxshare = max([pair[0] for pair in trnpairs])
minshare = min([pair[0] for pair in trnpairs])
if(features == "pn"):
    maxcount = max([pair[1] for pair in trnpairs])
    mincount = min([pair[1] for pair in trnpairs])
for trn_pair in trnpairs:
    trn_pair[0] = (trn_pair[0] - minshare) / (maxshare - minshare)
    if(features == "pn"):
        trn_pair[1] = (trn_pair[1] - mincount) / (maxcount - mincount)

# train MLE pdfs on each degree
import exact_mi_discrete as ex

# list of labeled mle_pdfs tuples [ ( label/degree,  [p/pn, prob] ) ]
labeled_mle_pdfs = []

# train MLE pdf on each type of relationship
for i in range(1, len(trn_inds)):
    beg = trn_inds[i - 1]
    end = trn_inds[i]
    prs = trnpairs[beg : end]
    
    # bin_N = int(len(prs) * 0.01)
    # special manual override
    bin_N = int(3000 * 0.01)
   
    if(features=="pn"):
        XX = np.array([[pair[0], pair[1]] for pair in prs])
        phi_binned = ex.bin_n(XX[:, 0], bin_N)
        XXb = [[phi_binned[i], XX[i][1]] for i in range(len(prs))]
       
        name = prs[0][2]
        pdf = ex.mle_pdf_vec(XXb)
    else:
        X = np.array([pair[0] for pair in prs])
        Xb = ex.bin_n(X, bin_N)
        
        name = prs[0][1]
        pdf = ex.mle_pdf_scal(Xb)
        
    labeled_mle_pdfs.append((name, pdf))

# export training information (& mle pdfs) to file (pipe on print output)
print(maxshare)
print(minshare)
if(features=="pn"):
    print(maxcount)
    print(mincount)
#print(len(labeled_mle_pdfs[0][1]))
print(labeled_mle_pdfs)