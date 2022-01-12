import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

# usage: python3 mi_estimator.py <directory with csv files> <target number of pairs / degree> <binning="lin"/"log"> <number of bins> <bin ratio (N * ratio) for degree-conditioned MI>
os.chdir(sys.argv[1])
target_pairs = int(sys.argv[2])
binning = sys.argv[3]
try:
    nobins = int(sys.argv[4])
except:
    print("using default number of bins")

try:
    nobins_ratio = float(sys.argv[5])
except:
    print("using default ratio of bins/datapoints")

'''
 mutual information MI(pairs ; degrees)
 pairs ~ X = [(phi, n)]
 degrees ~ y = [D]
'''
pairs = np.array([])
pairs_norm = np.array([])
degrees = np.array([])

deg_indices = [0]

for file in glob.glob("*.csv"):
    # get pairs from csv
    pairs_file = np.loadtxt(file, delimiter=',', usecols=(0, 1, 2))
    N = len(pairs_file)

    # all relationship-type csv files must be binned/concatenated with respect to degree
    for pair in pairs_file:
        if(len(pairs) == 0):
            pairs = np.array([pair[0:2]])
            pairs_norm = np.array([pair[0:2]])
            degrees = np.array(int(pair[2]))
        else:
            pairs = np.append(pairs, [pair[0:2]], axis=0)
            pairs_norm = np.append(pairs_norm, [pair[0:2]], axis=0)
            degrees = np.append(degrees, int(pair[2]))

    # get degree
    current_degree = int(pairs_file[0][2])
    # calculate number of pairs with no IBD sharing, add them into the simulation
    # times 2 for two rel. types in each deg
    if(target_pairs == -1):
        target_pairs = 80 * pow(2, current_degree - 1) # for exp distribution on 80
    elif(target_pairs == -2):
        target_pairs = int(500 * pow(2, (current_degree - 1) / 3)) # for slowexp distribution on 500

    deg_indices.append(deg_indices[-1] + target_pairs*2)
    number_zero_pairs = (target_pairs * 2) - N

    print("adding {} pairs with zero IBD sharing...".format(number_zero_pairs))
    for i in range(number_zero_pairs):
        pairs = np.append(pairs, [[0, 0]], axis=0)
        pairs_norm = np.append(pairs_norm, [[0, 0]], axis=0)
        degrees = np.append(degrees, current_degree)
    
    if(target_pairs == 80 * pow(2, current_degree - 1)):
        print("updating exp target")
        target_pairs = -1
    elif(target_pairs == int(500 * pow(2, (current_degree - 1) / 3))):
        print("updating slowexp target")
        target_pairs = -2
    
# normalize features of all training pairs
maxshare = max([pair[0] for pair in pairs])
maxcount = max([pair[1] for pair in pairs])
minshare = min([pair[0] for pair in pairs])
mincount = min([pair[1] for pair in pairs])
for pair in pairs_norm:
    pair[0] = (pair[0] - minshare) / (maxshare - minshare)
    pair[1] = (pair[1] - mincount) / (maxcount - mincount)

# separate features from pairs
phi_feature = np.array([pair[0] for pair in pairs_norm])
n_feature = np.array([pair[1] for pair in pairs_norm])

import exact_mi_discrete as ex
print("Exact MI(n ; D): {}".format(ex.ex_mi(n_feature, degrees)))

# binning method on MI
N = len(pairs_norm)
# for special manual number of bins (keeping bins constant across sample sizes, for example)
if 'nobins' not in globals():
    print("defaulting to N / 100 bins")
    nobins = int(N / 100)

if(binning == "lin"):
    phi_binned = ex.bin_n(phi_feature, nobins)
elif(binning == "log"):
    print("log no bins total: {}".format(nobins))
    phi_binned = ex.bin_logn(phi_feature, nobins)
else:
    print("defaulting to linear binning")
    phi_binned = ex.bin_n(phi_feature, nobins)

pairs_norm_binned = np.array(list(zip(phi_binned, n_feature)))

print("No. bins = {}".format(nobins))
print("Binned MI(phi ; D): {}".format(ex.ex_mi(phi_binned, degrees)))
print("Binned MI((phi, n) ; D): {}".format(ex.ex_mi_vec_scal(pairs_norm_binned, degrees)))

# calculate MI with decomposed joint probability
pdf_p_on_d = {}
pdf_n_on_d = {}
pdf_pn_on_d = {}

from sklearn.neighbors.kde import KernelDensity
# use un-normalized features for mi conditioned on degree (leave n discrete)
for i in range(1, len(deg_indices)):
    beg = deg_indices[i - 1]
    end = deg_indices[i]
    deg_pairs = pairs_norm[beg : end]
    deg = degrees[beg]
    
    feat1 = [dpair[0] for dpair in deg_pairs]
    print("len: {}".format(len(feat1)))
    
    if "nobins_ratio" in globals():
        nobins_d = int(len(feat1) * nobins_ratio)
    else:
        print("defaulting to N/150 bins")
        nobins_d = int(len(feat1) / 150)

    print("Number of bins (in deg. {}): {}".format(deg, nobins_d))

    if(binning == "lin"):
        feat1_binned = ex.bin_n(feat1, nobins_d)        
    elif(binning == "log"):
        # print("log no bins: {}".format(nobins_d))
        # feat1_binned = ex.bin_logn(feat1, nobins_d)
        # does it make sense to logarithmically bin a single degree?
        # the rationale for log binning is that smaller degrees
        #   get larger bin sizes -- goes away for a single degree
        print("single degree: no log binning")
        feat1_binned = ex.bin_n(feat1, nobins_d)
    else:
        print("defaulting to linear binning (conditioned on D)")
        feat1_binned = ex.bin_n(feat1, nobins_d)
    
    feat1_binned = feat1_binned.tolist()
    feat2 = [dpair[1] for dpair in deg_pairs]
    # mi = mi_knear(feat1, feat2, 3)    
    
    # train MI on uniform set, then evaluate on restricted (exponential) data
    #pdfP = ex.mle_pdf_scal(feat1)
    pdfP = ex.mle_pdf_scal(feat1_binned)
    
    pdfN = ex.mle_pdf_scal(feat2)
    
    #pdfPN = ex.mle_pdf_vec(deg_pairs)
    deg_pairs_binned = np.array(list(zip(feat1_binned, feat2)))
    pdfPN = ex.mle_pdf_vec(deg_pairs_binned)
  
    # change pdfs to dictionaries
    pdfPdict = {p[0] : p[1] for p in pdfP}
    pdfNdict = {n[0] : n[1] for n in pdfN}
    pdfPNdict = {(pn[0], pn[1]) : pn[2] for pn in pdfPN}

    # update conditional pdfs (for decomposed MI computation after loop)
    pdf_p_on_d.update({deg : pdfPdict})
    pdf_n_on_d.update({deg : pdfNdict})
    pdf_pn_on_d.update({deg : pdfPNdict})
    
    expsize = int(160 * pow(2, deg - 1))
    feat1_restrict = feat1_binned[:expsize]
    feat2_restrict = feat2[:expsize]
    #print("calculating MIs exactly...")
    
    ex_mi = ex.ex_mi(feat1_binned, feat2)
    ex_mi_interp = ex.mi_from_exact_pdfs(pdfPdict, pdfNdict, pdfPNdict, feat1_binned, feat2)
    ex_mi_restrict_interp = ex.mi_from_exact_pdfs(pdfPdict, pdfNdict, pdfPNdict, feat1_restrict, feat2_restrict)
    ex_mi_restrict = ex.ex_mi(feat1_restrict, feat2_restrict)

    # number per bin
    print("size (p) restricted from {} to {}".format(len(feat1_binned), len(feat1_restrict)))
    print("size (n) restricted from {} to {}".format(len(feat2), len(feat2_restrict)))
    #print("Binned MI of degree {}: {}".format(deg, ex_mi))
    print("Binned MI of degree(uni)(interp) {}: {}".format(deg, ex_mi_interp))
    print("Binned MI of degree(uni)(exact) {}: {}".format(deg, ex_mi))
    print("Binned MI of degree(exp)(exact) {}: {}".format(deg, ex_mi_restrict))
    print("Binned MI of degree(exp)(interp) {}: {}".format(deg, ex_mi_restrict_interp))

# decomposed MI
pdf_d_uni = {d : (1/7) for d in range(1, 8)}
print("distribution pdf uni: {}".format(pdf_d_uni))

distribution_exp = np.array([160 * pow(2, d - 1) for d in range(1, 8)])
pdf_d_exp = distribution_exp / sum(distribution_exp)
pdf_d_exp = {d : pdf_d_exp[d-1] for d in range(1, 8)}
print("distribution pdf exp: {}".format(pdf_d_exp))

distribution_slowexp = np.array([1000 * pow(2, (d - 1)/3) for d in range(1, 8)])
pdf_d_slowexp = distribution_slowexp / sum(distribution_slowexp)
pdf_d_slowexp = {d : pdf_d_slowexp[d-1] for d in range(1, 8)}
print("distribution pdf slowexp: {}".format(pdf_d_slowexp))

distpdfs = [pdf_d_uni, pdf_d_exp, pdf_d_slowexp]
distnames = ["uniform", "exponential", "slowexp"]
for i in range(len(distpdfs)):
    d_name = distnames[i]
    d_pdf = distpdfs[i]
    print("Decomposed ({}) MI(n ; D): {}".format(d_name,
                                                 ex.decomposed_MI(pdf_n_on_d, 
                                                                  d_pdf,
                                                                  n_feature,
                                                                  degrees)))
    print("Decomposed ({}) MI(p ; D): {}".format(d_name,
                                                 ex.decomposed_MI(pdf_p_on_d, 
                                                                  d_pdf,
                                                                  phi_binned,
                                                                  degrees)))
    print("Decomposed ({}) MI((p, n) ; D): {}".format(d_name, 
                                                      ex.decomposed_MI(pdf_pn_on_d, 
                                                                       d_pdf,
                                                                       pairs_norm_binned,
                                                                       degrees)))