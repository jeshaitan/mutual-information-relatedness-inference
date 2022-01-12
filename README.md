# mutual-information-relatedness-inference
### Tools for calculating mutual information of identical-by-descent statistics; Bayesian degree-of-relatedness classification.

`ibd_props.py` outputs csv-formatted pairwise IBD proportion and segment number given `.seg` files.

`mi_estimator.py` uses `exact_mi_discrete.py` tools to calculate MIs on the csv pairwise populations.

`char_trn.py` and `char_tst.py` train and test a Bayesian degree-of-relatedness classifier on the pairwise populations.
