# mutual-information-relatedness-inference
### Tools for calculating mutual information of identical-by-descent statistics; Bayesian degree-of-relatedness classification.

#### `ibd_props.py` outputs csv-formatted pairwise IBD proportion and segment number given `.seg` and SIMMAP files.
- **Usage:** `python3 ibd_props.py <output.seg file> <simmap file>`
- **Pairwise CSV output:** `<IBD proportion>,<IBD segment number>,<known degree of relatedness>,<name of pair>\n`

#### `mi_estimator.py` uses `exact_mi_discrete.py` tools to calculate MIs on the csv pairwise populations.
- **Usage:** `python3 mi_estimator.py <directory with csv files> <target number of pairs / degree> <binning="lin"/"log"> <number of bins> <bin ratio (N * ratio) for degree-conditioned MI> > output` with shell redirection for outputting.
    - CSV file directory must be formatted such that there is one unique file per degree of relatedness: _e.g._ `deg_1.csv, deg_2.csv, ...`
    - `mi_estimator` will pad each degree with 0.0 IBD sharing pairs to meet the target number of pairs per degree to take into account those zero-sharing pairs in those degrees.
    - Logarithmic binning bins more densely at lower sharing, _meant to_ capture closely packed (in IBD prop. and seg. number space) higher degrees.
    - `<number of bins>` is across all degrees, whereas `<bin ratio...>` specifies bins per degree for outputing MI conditioned on _D_.
- **MI output:**  
    - number of padded (zero IBD sharing) pairs per degree
    - total number of pairs
    - number of bins per degree
    - number of pairs per degree
    - uniform, exponential-growth, and slow exponential-growth distributions
    - MI(_F_ ; _D_) for the uniform, exponential-growth, and slow exponential-growth (pairwise restricted) versions of the input pairs

#### `char_trn.py` and `char_tst.py` train and test a Bayesian degree-of-relatedness classifier on the pairwise populations.
##### `char_trn.py`
- **Usage:** `python3 char_trn.py <training csvs/> <features: "p", "n", "pn"> <target training number> > training_info` with shell redirection for outputting
    - CSV file directory should again by formatted by file by degree: _e.g._ `train_deg_1.csv, train_deg_2.csv, ...`
    - `<target training number>` specifies _total_ training number across all degrees.
- **Training information output:** 
    - Maximum and minimum IBD sharing in training data (`features="p", "pn"`)
    - Maximum and minimum IBD segment number count in training data (`features="n", "pn"`)
    - MLE PDFs reporting probabilities of realizations of _F_

##### `char_tst.py`
- **Usage:** `python3 char_tst.py <testing csvs/> <features: "p", "n", "pn"> <training output> <training output (p) as fallback> > sensitivities` with shell redirection for outputting
    - `<training output>` should here be the entire output file of `char_trn.py`.
    - Should also provide training output trained on "p" for fallback probabilities on missing "n" (and therefore "pn") realizations.
- **Classification output:**
    - Accuracy and incorrect classification counts for each degree
    - Checksums on number of pairs and number of classified pairs.
    - Number of unclassified pairs (correct and missed unrelated pairs)
    - Uncommenting the last section of `char_tst.py` generates histograms of correct and incorrect classified pairs in each degree, plotted along "p" and "n" realizations.

#### `exact_mi_discrete.py` contains tools for calculating MI from different combinations of sources, vectors, and PDFs, as well as binning and interpolating functions used in the other codes. 
