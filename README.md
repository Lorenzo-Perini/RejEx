# RejEx: Rejecting via ExCeeD

`RejEx` (Rejecting via ExCeeD) is a GitHub repository containing the **RejEx** [1] algorithm.
It refers to the paper titled *Unsupervised Anomaly Detection with Rejection* published at NeurIPS 2023.

Check out the NeurIPS paper here: [ArXiv](https://arxiv.org/pdf/2305.13189.pdf) and [OpenReview](https://openreview.net/pdf?id=WK8LQzzHwW).

## Abstract

Anomaly detection aims at detecting unexpected behaviours in the data. Because anomaly detection is usually an unsupervised task, traditional anomaly detectors learn a decision boundary by employing heuristics based on intuitions, which are hard to verify in practice. This introduces some uncertainty, especially close to the decision boundary, that may reduce the user trust in the detector’s predictions. A way to combat this is by allowing the detector to reject examples with high uncertainty (*Learning to Reject*). This requires employing a confidence metric that captures the distance to the decision boundary and setting a rejection threshold to reject low-confidence predictions. However, selecting a proper metric and setting the rejection threshold without labels are challenging tasks. In this paper, we solve these challenges by setting a constant rejection threshold on the stability metric computed by **ExCeeD**. Our insight relies on a theoretical analysis of such a metric. Moreover, setting a constant threshold results in strong guarantees: we estimate the test rejection rate, and derive a theoretical upper bound for both the rejection rate and the expected prediction cost. Experimentally, we show that our method outperforms some metric-based methods.

## Contents and usage

The repository contains:
- RejEx.py, a function that allows to get 1) the confidence values, 2) the predictions with rejection, and 3) the theoretical resutls (i.e., rejection rate, and the upper bounds on the rejection rate and the test cost);
- Notebook.ipynb, a notebook showing how to use RejEx on an artificial dataset;

To use RejEx, import the github repository or simply download the files. You can also find the benchmark datasets at this [[link](https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/)].


## Rejecting via ExCeeD (RejEx)

Given a dataset with attributes **X**, an unsupervised anomaly detector assigns to each example an anomaly score, representing its degree of anomalousness. **ExCeeD** tranforms the anomaly scores into stability-based confidence values in a detector-agnostic fashion. Given the sensitivity value T (default = 32), RejEx rejects all examples with stability lower than 1-2exp(-T).

Specifically, given a training set **X_train** to learn a detector **ad**, the contamination factor **contamination**, the sensitivity value **T**, and a test dataset **X_test**, the algorithm is applied as follows:

```python
from pyod.models.iforest import IForest
from RejEx import predictions_with_rejection

# Train an anomaly detector (for instance, here we use IForest)
contamination = 0.1
ad = IForest(contamination = contamination).fit(X_train)

# Compute the predictions with rejection
T=32
predictions_with_rejection = predict_with_RejEx(ad, X_test, T, contamination)
```

Evaluating a model with rejection is a non-trivial task. In the paper, we evaluate the detectors by using an additive cost function. Given three costs **c_fp** (false positives), **c_fn** (false negatives), and **c_r** (rejection) and the true test labels **y_test**

```python
from RejEx import evaluate_model_with_rejection

# Evaluate the detector with rejection
cost_with_rejection = evaluate_model_with_rejection(predictions_with_rejection, y_test, c_fp, c_fn, c_r)
```

Finally, in the paper we showed that we can estimate at training time three relevant properties: 1) the expected rejection rate, 2) an upper bound for the rejection rate, 3) an upper bound for the test cost. Fixed the desired probability **1-delta**, you can obtain these values by running:

```python
from RejEx import *

n_train = len(X_train)
n_test = len(y_test)
delta = 0.10

estimate_rejection_rate = expected_rejection_rate(n_train, contamination, T)
upper_bound_rejection_rate = get_upper_bound_rr(n_train, contamination, T, delta)
upper_bound_expected_cost = upperbound_cost(n_train, contamination, T, c_fp, c_fn, c_r)
```

## Dependencies

The `RejEx` function requires the following python packages to be used:
- [Python 3](http://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](http://www.scipy.org)
- [Pandas](https://pandas.pydata.org/)


## Contact

Contact the main author of the paper: [lorenzo.perini@kuleuven.be](mailto:lorenzo.perini@kuleuven.be).


## References

[1] Perini, L., Davis, J.: *Unsupervised Anomaly Detection with Rejection*. Advances in Neural Information Processing Systems, 2023.
