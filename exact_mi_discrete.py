import numpy as np

'''
 calculates the exact MI of two discrete random variables X & Y
 with realizations x & y
 
 X and Y are numpy arrays
'''
def ex_mi(X, Y):
    assert len(X) == len(Y)

    # summation indices and counts of each realization
    indx, count_x = np.unique(X, return_counts=True)
    indy, count_y = np.unique(Y, return_counts=True)

    Nx = len(X)
    Ny = len(Y)
    MI = 0
    for i in range(len(indx)):
        px = count_x[i] / Nx
        for j in range(len(indy)):
            py = count_y[j] / Ny
            
            acc_xy = 0
            for k in range(Nx):
                if(X[k] == indx[i] and Y[k] == indy[j]):
                    acc_xy += 1
            p_xy = acc_xy / Nx

            # do not compute log of zero
            if(p_xy == 0):
                continue
                
            MI += p_xy * np.log(p_xy / (px * py))

    return MI

'''
 calculates MI from given pdfs, but without interpolation
 used where MI(same data as pdfs fit to)
 x and y must be scalar RVs
 pdfs must be given as dictionaries (float keys), pdfXY has tuple keys
'''
def mi_from_exact_pdfs(pdfXdict, pdfYdict, pdfXYdict, X, Y):
    assert len(X) == len(Y)
    
    # get realizations of X and Y
    xs = np.unique(X)
    ys = np.unique(Y)

    MI = 0
    for x in xs:
        px = pdfXdict[x]
        for y in ys:
            py = pdfYdict[y]        

            try:
                # not every combination is valid
                p_xy = pdfXYdict[(x, y)]
            except KeyError as e:
                continue
            
            MI += p_xy * np.log(p_xy / (px * py))
    
    return MI

'''
 MI(X;Y) = sum_x{ sum_y{ 
       p(x|y) * p(y) * log( p(x|y) / sum_y'{ p(x|y')p(y') } ) 
    } }
 pdfXonYdicts is dictionary of dictionary-pdfs, 
    one pdf for (conditioned on) each degree

 if only one statistic is multivariate use X for that statistic ==> MI (x=[(p, n)], y=[d])  
'''
def decomposed_MI(pdfXonYdicts, pdfYdict, X, Y):
    assert len(X) == len(Y)
    
    # when running for MI(X=(feats) ; Y=D)
    assert len(pdfXonYdicts) == 7
    
    # get realizations of X and Y
    xs = np.unique(X, axis=0)
    ys = np.unique(Y)

    # make vector type X hashable
    if(np.shape(xs) == (len(xs),2)):
        xs = [(x[0], x[1]) for x in xs]

    MI = 0
    # iterate through phis or (phi, n)s
    for x in xs:
        # denominator in log is constant throughout summation in D
        denom = sum([getmaybe(pdfXonYdicts[yprime], x) * pdfYdict[yprime]
                     for yprime in ys])

        # iterate through degrees (1..7)
        for y in ys:
            # retrieve prob in yth degree pdf
            pxony = getmaybe(pdfXonYdicts[y], x)
            py = pdfYdict[y]

            # dont compute log of zero
            if(pxony == 0):
                continue

            MI += pxony * py * np.log(pxony / denom)
            
    return MI

'''
 if value is not in pdf, return 0.0 score
'''
def getmaybe(pdfdict, key):
    try:
        return pdfdict[key]
    except KeyError as e:
        return 0.0

'''
 calculates MI with two given pdfs of RVs X and Y and their joint probability
 only for two scalars -- MI(p, n)
'''
def mi_from_pdfs(pdfX, pdfY, pdfXY, X, Y):
    assert len(X) == len(Y)
    
    # get realizations of X and Y
    xs = np.unique(X)
    ys = np.unique(Y)

    MI = 0
    for x in xs:        
        px = interpolate_score(x, pdfX, False)
        for y in ys:
            py = interpolate_score(y, pdfY, False)        

            p_xy = interpolate_score([x, y], pdfXY, True)
            if(p_xy == 0):
                continue
            
            MI += p_xy * np.log(p_xy / (px * py))
    
    return MI

'''
 calculates the joint pdf of RVs X & Y
 returns list of [value_x, value_y, probability] pairs
'''
def mle_pdf_joint(X, Y):

    valx, count_x = np.unique(X, return_counts=True)
    valy, count_y = np.unique(Y, return_counts=True)

    Nx = len(X)
    pdf = []
    for i in range(len(valx)):
        for j in range(len(valy)):
            acc_xy = 0
            for k in range(Nx):
                if(X[k] == valx[i] and Y[k] == valy[j]):
                    acc_xy += 1
            p_xy = acc_xy / Nx
            pdf.append([valx[i], valy[j], p_xy])
    
    return pdf
 
'''
 calculates the pdf of a discrete RV, X
 returns list of [value, probability] pairs
'''
def mle_pdf_scal(X):

    valx, count_x = np.unique(X, return_counts=True)
    
    Nx = len(X)
    pdf = []
    for i in range(len(valx)):
        p_i = count_x[i] / Nx
        pdf.append([valx[i], p_i])
    
    return pdf

'''
 calculates the pdf of discrete RV, XX
 with vector realization xx
 returns list of [xx[0], xx[1], probability] pairs
'''
def mle_pdf_vec(XX):

    valxx, count_xx = np.unique(XX, return_counts=True, axis=0)
    
    Nxx = len(XX)
    pdf = []
    for i in range(len(valxx)):
        p_i = count_xx[i] / Nxx
        pdf.append([valxx[i][0], valxx[i][1], p_i])
    
    return pdf

'''
 decompose P(p , n) into P(p | n)P(n)
'''
# TODO

from scipy import array

'''
 get interpolated score from pdf (smoothing)
'''
from scipy.interpolate import griddata, interp1d, interp2d
from math import isnan
def interpolate_score(sample, mle_pdf, is_vec):
    
    pdf = np.array(mle_pdf)
    xi = np.array(sample)
    
    # two dimensional interpolation (griddata vs interp2d)
    if(is_vec):
        # inter = griddata(pdf[:, 0:2], pdf[:, -1], xi, method='linear')[0]
        f_in = interp2d(pdf[:, 0], pdf[:, 1], pdf[:, 2], kind='cubic')
        inter = f_in(sample[0], sample[1])
        if(isnan(inter)):
            # for now, return zero if outside of interpolated volume
#            print('sending zero... 2d')
            return 0.0
        else:
            return inter
    # one dimensional interpolation
    else:
        try:
            f_in = interp1d(pdf[:, 0], pdf[:, -1])
            inter = f_in(sample)
            return inter
        except ValueError as e:
            #print('error: {}'.format(e))
            #print('sending zero 1d...')
            return 0.0
'''
 return interpolated score given interp<1,2>d function
 do not return negative value
'''        
def interpscore(sample, f_interp, feats):
    if(feats=="pn"):
        try:
            inter = f_interp(sample[0], sample[1])
            return max(0, inter)
        except ValueError as e:
            return 0.0
    else:
        try:
            inter = f_interp(sample)
            return max(0, inter)
        except ValueError as e:
            return 0.0

'''
 calculates the exact MI of two discrete RVs XX & Y
 XX has a vector realization, Y is a scalar

 XX is a numpy matrix of shape (n, 2), Y has shape (n, 1)
'''
def ex_mi_vec_scal(XX, Y):
    assert len(XX) == len(Y)

    # summation indices and counts of each realization
    indxx, count_xx = np.unique(XX, return_counts=True, axis=0)
    indy, count_y = np.unique(Y, return_counts=True)

    Nxx = len(XX)
    Ny = len(Y)
    MI = 0

    for i in range(len(indxx)):
        pxx = count_xx[i] / Nxx
        for j in range(len(indy)):
            py = count_y[j] / Ny
            
            acc_xxy = 0
            for k in range(Nxx):
                if(np.all(XX[k] == indxx[i]) and Y[k] == indy[j]):
                    acc_xxy += 1
            p_xxy = acc_xxy / Nxx

            # do not compute log of zero
            if(p_xxy == 0):
                continue
                
            MI += p_xxy * np.log(p_xxy / (pxx * py))
    
    return MI
    
'''
 returns a binned copy of the X
 where N is the number of bins
 
 side-effect: prints a list of number-of-datapoints in each bin

 X is a numpy array
'''
import itertools
import matplotlib.pyplot as plt
def bin_logn(X, N):
    # boundary value below the smallest non-zero value
    epsilon = np.min(X[np.nonzero(X)])
    
    bins = np.logspace(np.log2(epsilon), np.log2(max(X)), num=N, base=2.0)
    # bins[0] = 0.0
    
    indices = np.digitize(X, bins, right=True)
    binnedX = np.array([bins[ind] for ind in indices])
    
    plt.plot(binnedX)                  
    plt.savefig('unsorted_log_phi.png')
    
    # prints number of elements in each bin
    # print([(elem, len(list(group))) for elem, group in itertools.groupby(sorted(binnedX))])
    return binnedX
    
def bin_n(X, N):
    bins = np.linspace(min(X), max(X), num=N)
    indices = np.digitize(X, bins, right=True)
    binnedX = np.array([bins[ind] for ind in indices])                                     
                                                  
    plt.plot(binnedX)                                                                      
    plt.savefig('unsorted_lin_phi.png')

    # prints number of elements in each bin                                                    
    # print([(elem, len(list(group))) for elem, group in itertools.groupby(sorted(binnedX))])
    return binnedX
