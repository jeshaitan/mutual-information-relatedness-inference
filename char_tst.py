import numpy as np
import matplotlib.pyplot as plt
import sys
import glob, os

#usage: python3 char_tst.py <testing csvs/> <features: "p", "n", "pn"> <training output> <training output (p) as fallback>
os.chdir(sys.argv[1])
features = sys.argv[2]

targettest = 21000

# load normalization info and pdfs from training data
'''
maxshare
minshare
(maxcount
mincount)
[ (degree, [p, (n,) prob]) ]
'''
with open(sys.argv[3]) as info:
    lines = info.readlines()
    maxshare = float(lines[0])
    minshare = float(lines[1])
    if(features=="pn"):
        maxcount = int(float(lines[2]))
        mincount = int(float(lines[3]))
        labeled_mle_pdfs = eval(lines[4])
    else:
        labeled_mle_pdfs = eval(lines[2])


# thresholds for classifying unrelated pairs
# thresh_share = minshare
thresh_share = 0.00276
thresh_count = mincount

# pairs should be sorted by degree
tstpairs = []
tstpairs_labels = []
tst_inds = [0]
total_unrel = 0
for tstfile in glob.glob("*.csv"):
    tstpairs_i = np.loadtxt(tstfile, delimiter=',', usecols=(0, 1, 2))
    name = tstpairs_i[0][2]
    
    tstsize_i = len(tstpairs_i)
    num_unrel_i = 0
    for pair in tstpairs_i[0 : tstsize_i]:
                
        if(features == "pn"):
            if(pair[0] <= thresh_share):
                num_unrel_i += 1
            else:
                tstpairs_labels.append(pair[2])
                tstpairs.append(np.delete(pair, 2))
        elif(features == "p"):
            if(pair[0] <= thresh_share):
                num_unrel_i += 1
            else:
                tstpairs_labels.append(pair[2])
                tstpairs.append(np.delete(pair, [1, 2]))
        elif(features == "n"):
            if(pair[1] <= thresh_count):
                num_unrel_i += 1
            else:
                tstpairs_labels.append(pair[2])
                tstpairs.append(np.delete(pair, [0, 2]))
    total_unrel += num_unrel_i
    tstsize_i -= num_unrel_i
    tst_inds.append(tst_inds[-1] + tstsize_i)

print("Total classified as unrelated pairs: {}".format(total_unrel))

#assert len(tstpairs) == targettest

# read in fallpack (p) pdf
if(features=="pn"):
    with open(sys.argv[4]) as fallback_info:
        lines = fallback_info.readlines()
        fb_labeled_mle_pdfs = eval(lines[2])

# IBIS Degree thresholds (to be normalized)
thresholds = [0.475, 0.176776695, 0.088388348, 0.044194174, 0.022097087, 0.011048543, 0.005524272, 0.002762136]

'''
# normalize all pairs according to testing.
maxshare = max([pair[0] for pair in tstpairs])
minshare = min([pair[0] for pair in tstpairs])
if(features == "pn"):
    maxcount = max([pair[1] for pair in tstpairs])
    mincount = min([pair[1] for pair in tstpairs])
'''

# normalize all data according to min&max of training
for tst_pair in tstpairs:
    tst_pair[0] = (tst_pair[0] - minshare) / (maxshare - minshare)
    if(features == "pn"):
        tst_pair[1] = (tst_pair[1] - mincount) / (maxcount - mincount)
for t in range(len(thresholds)):
    thresholds[t] = (thresholds[t] - minshare) / (maxshare - minshare)

import exact_mi_discrete as ex

from scipy.interpolate import griddata, interp1d
from math import isnan

# score samples under each mle_pdf
def scoresamples(lmlepdfs, scorefeats):
    mle_like_list = []
    for (label, mle) in lmlepdfs:
        scores = []

        pdf = np.array(mle)
        xi = np.array(tstpairs)

        if(scorefeats=="pn"):
            gd_score_linear = griddata(pdf[:, 0:2], pdf[:, -1], xi, method='linear')
        else:
            f_in = interp1d(pdf[:, 0], pdf[:, -1]) 
        for smpl in range(len(tstpairs)):
            # add score based on linear interpolation of probabilities
            # scores beyond range (only one nearest neighbor) get zero score

            # if multivariate, pick best interpolated score
            if(scorefeats=="pn"):
                g_score = gd_score_linear[smpl]
                # make sure score is positive
                if(isnan(g_score)):
                    scores.append(0.0)                
                else:
                    scores.append(g_score)
            else:
                sample = tstpairs[smpl][0]
                i_score = ex.interpscore(sample, f_in, scorefeats)
                scores.append(i_score)
            
        mle_like_list.append((label, scores))
    return mle_like_list

mle_like_list = scoresamples(labeled_mle_pdfs, features)
if(features=="pn"):
    # create fallback scores using (p) pdfs
    fb_mle_like_list = scoresamples(fb_labeled_mle_pdfs, "p")

'''sum log probabilities accounting for possible underflows'''
def sumLogProbs(a, b):
    if a > b:
        return a + np.log1p(np.exp(b - a))
    else:
        return b + np.log1p(np.exp(a - b))

def sumlps(lps):
    lps_fin = np.select([np.array(lps) != np.NINF], [np.array(lps)], default=0)
    acc = lps_fin[0]
    for i in range(1, len(lps_fin)):
        acc = sumLogProbs(acc, lps_fin[i])
    return acc

'''
 returns list of final labels given a list of [(label1, scores), (label2, scores), ...]
'''
def pickfromscores(lklist):

    # final labels for each test pair 
    categories = np.zeros(len(tstpairs))

    scores_flat = np.array(lklist, dtype=object)[:, 1]
    # unclassified in (p, n): switch to fallback pdf in (p)
    if(features=="pn"):
        fb_lklist_dict = dict(fb_mle_like_list)
        for t in range(len(tstpairs)):
            if(sum([s[t] for s in scores_flat]) == 0):           
                # replace column of scores for t with fallback scores from (p)
                for scorerow in range(len(lklist)):
                    label = lklist[scorerow][0]
                    lklist[scorerow][1][t] = fb_lklist_dict[label][t]
 
    # mark unclassified pairs (if pn, /still/ unclassified pairs)
    #checksm=0
    for t in range(len(tstpairs)):
        if(sum([s[t] for s in scores_flat]) == 0):           
            categories[t] = -1
            continue
        '''
        # unrelated pair (after normalization, these pairs will have negative sharing)
        if(features == "p" or features == "pn"):
            if(tstpairs[t][0] <= minshare):
                checksm+=1
                categories[t] = -1
                continue
        else:
            if(tstpairs[t][1] < mincount):
                checksm+=1
                categories[t] = -1
                continue
        '''
    #print(checksm)

    for i in range(len(lklist)):
        # take logarithm of likelihoods
        lklist[i] = (lklist[i][0], np.log(lklist[i][1]))

    for i in range(len(lklist[0][1])):
        # denominator p(x): sum{p(D) * p(x|D)}
        sum_cats = sumlps([(lk[1][i]) for lk in lklist])
        #sum_cats = sumlps([(logprior[lk[0]] + lk[1][i]) for lk in lklist])

        for lk in lklist: 
            # multiply by the bayesian prior to get a posterior 
            #lk[1][i] += logprior[lk[0]]
            
            # normalize the likelihoods by the probability of being in any distribution
            lk[1][i] -= sum_cats

    # pick the maximum score list
    for t in range(len(tstpairs)):
        if(categories[t] == -1):
           continue
        
        max_l = lklist[0][1][t]
        categories[t] = lklist[0][0]
        
        for l in lklist:
            if l[1][t] > max_l:
                max_l = l[1][t]
                categories[t] = l[0]
                assert l[0] != 0.0

    return categories


# decide categories under either the kde scores or the MLE scores
def dotest(like_list, tst_list, tst_labels):
    
    categories = pickfromscores(like_list)
    colors = list(range(1,8))

    # score accuracy (number of correct categorizations out of total)
    # for these lists, the degree = index + 1 (e.g. degree 1 is index 0)
    correct_in_degree = np.zeros(len(colors))
    number_in_degree = np.zeros(len(colors))

    correct_unrelated = 0
    missed_unrelated = np.zeros(len(range(1,8)))
    
    #count up the incorrectly labeled pairs in each degree
    missed_degs = {d : {d_wrong : 0 
                        for d_wrong in range(1, 8) if d_wrong != d} 
                   for d in range(1, 8)}

    # data for histograms of classifications
    hists = {deg_of_hist : [] for deg_of_hist in range(1, 8)}
    scoresbylabel = np.array(like_list, dtype=object)
    labels = scoresbylabel[:, 0]
    scores_l = scoresbylabel[:, 1]
    unclassified_by_deg = {d : 0 for d in range(1, 8)}
    checksum = [0,0,0,0,0]
    for c in range(len(categories)):
        if(categories[c] == tst_labels[c]):
            if(categories[c] == -1):
                # correctly classified as incorrect
                correct_unrelated += 1
                checksum[0] += 1
            else:
                # count a correct classification for the degree
                correct_in_degree[int(tst_labels[c]) - 1] += 1
                checksum[1] += 1
        elif(categories[c] == -1):
            # count a data point that was unable to be classified (0.0 7-way ties)
            unclassified_by_deg[int(tst_labels[c])] += 1
            checksum[2] += 1
        else:
            # truly unrelated, given label (categories != 0)
            if(tst_labels[c] == -1):
                missed_unrelated[int(categories[c]) - 1] += 1
                checksum[3] += 1
            else:
                # count an incorrect classification for the degree, mark in missed_degs
                missed_degs[int(tst_labels[c])][categories[c]] += 1
                checksum[4] += 1
        '''
        # mark the classification given into the particular realization of feature
        if(categories[c] != -1): # do not draw unclassified points
            hists[tst_labels[c]].append((tst_list[c], categories[c]))
        '''
        
        # record a number only for truly related pair
        if(tst_labels[c] != -1):
            number_in_degree[int(tst_labels[c]) - 1] += 1
    
    for c in range(len(colors)):
        accuracy = correct_in_degree[c] / number_in_degree[c]
        print("Degree {} accuracy: {} ({})".format(c + 1, accuracy, correct_in_degree[c]))
        print("Degree {} incorrects: {}".format(c + 1, missed_degs[c + 1]))
    
    # check sum 
    print("Total: {}, checksum total: {}".format(len(categories), sum(checksum)))
    print("Checksum: {}".format(checksum))
    print("Unclassified: {}".format(unclassified_by_deg))
    print("Correct unrelated: {}".format(correct_unrelated))
    print("Missed unrelated: {}".format(missed_unrelated))
    print("{} relatives total".format(sum(number_in_degree)))
    
    '''
    # generate histograms
    fig, ax = plt.subplots(7, figsize=(10, 20), dpi=600, constrained_layout=True)

    def statement(d, i, z):
        if(d == i):
            return "Deg. {}".format(i)
        elif(z == []):
            return "_D"
        else:
            return "Deg. {} ({})".format(i, missed_degs[d][i])

    for d in range(1, 8):
        # split and plot data by categories
        all_realizations = [[] for d_classified in range(0, 7)]

        for point in hists[d]:
            if(features=="pn"):
                # [0][0] for cross-section in phi, [0][1] for cross-section in n
                all_realizations[int(point[1] - 1)].append(point[0][0])
            else:
                all_realizations[int(point[1] - 1)].append(point[0])
	
        # print in simple csv format degree,accuracy,unclassified
        # print("{},{},{}".format(d, correct_in_degree[d - 1] / number_in_degree[d - 1], unclassified_by_deg[d]))        
        ax[d - 1].title.set_text("Deg. {} (Accuracy: {}) (Unclassified: {})".format(d, correct_in_degree[d - 1] / number_in_degree[d - 1], unclassified_by_deg[d]))
        ax[d - 1].hist(all_realizations, bins=np.arange(0, 1 + 0.001, 0.001), stacked=True)
        ax[d - 1].set_xticks(np.arange(0, 1 + 0.01, 0.01))
        plt.setp(ax[d - 1].get_xticklabels()[::2], visible=False)
        ax[d - 1].tick_params(axis='x', labelsize='small', labelrotation=90)
        non_empty_degs = [statement(d, i, all_realizations[i - 1]) for i in range(1, 8)]
        ax[d - 1].legend(non_empty_degs)

        # add degree threshold lines from IBIS
        for t in range(1, len(thresholds)):
            #print(thresholds[t])
            if(t == 7):
                ax[d - 1].axvline(x=thresholds[t], c='k')
            else:
                ax[d - 1].axvline(x=thresholds[t], c='k')        
        
        fig.suptitle("Features: ({})".format(features))
        plt.savefig('real_trnnorm_hists_{}.png'.format(features))
        plt.show()                
    '''

dotest(mle_like_list, tstpairs, tstpairs_labels)