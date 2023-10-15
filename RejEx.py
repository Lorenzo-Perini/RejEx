import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.optimize import root_scalar

######################################################################################
# The following functions have been used for the experimental comparison of the paper
#                 Unsupervised Anomaly Detection with Rejection
######################################################################################

def predict_with_RejEx(clf, X_test, T, contamination):
    rejection_thr = 1- 2*np.exp(-T)
    confidence, test_psi_n = pred_confidence(clf, X_test)
    y_pred = clf.predict(X_test)
    y_pred[np.where(confidence<=rejection_thr)[0]] = -2
    return y_pred
    
    
def pred_confidence(clf, X):
    # This function returns ***the model confidence*** using ExCeeD's stability, as implemented in PyOD.
    n = len(clf.decision_scores_)
    test_scores = clf.decision_function(X)
    count_instances = np.vectorize(lambda x: np.count_nonzero(clf.decision_scores_ <= x)) 
    n_instances = count_instances(test_scores)
    posterior_prob = np.vectorize(lambda x: (1+x)/(2+n))(n_instances)
    stability = np.vectorize(lambda p: 1-binom.cdf(n-np.int(n*clf.contamination),n,p))(posterior_prob)
    confidence = 2*abs(stability-.5)
    psi_n = n_instances/n
    return confidence, psi_n

def expected_rejection_rate(n, cont, T):
    # This function returns ***our estimate of the rejection rate*** at test time using only
    # training values of #examples (n), contamination factor (cont), and tolerance error (T).
    ngammaminus1 = int(n * cont) -1
    argsmin = (ngammaminus1, n, 1-np.exp(-T))
    argsmax = (ngammaminus1, n, np.exp(-T))
    q1 = root_scalar(Fmin, bracket=[0, 1], method='brentq', args = argsmin).root
    q2 = root_scalar(Fmin, bracket=[0, 1], method='brentq', args = argsmax).root
    return q2-q1

def get_upper_bound_rr(n, g, T, delta):
    # This function returns ***the upper bound for the rejection rate*** that is satisfied
    # with probability 1-delta.
    right_mar=(-g*(n+2)+n+1)/n +(T*(n+2))/(np.sqrt(2*n**3*T))
    right_mar = min(1, right_mar)
    left_mar=(2+n*(1-g)*(n+1))/n**2-np.sqrt(0.5*n**5*(2*n*(-3*g**2-2*n*(1-g)**2+4*g-3)+T*(n+2)**2-8))/n**4
    left_mar = max(0,left_mar)
    add_term = 2*np.sqrt(np.log(2/delta)/(2*n))
    return right_mar - left_mar + add_term

def upperbound_cost(n, cont, T, c_fp, c_fn, c_r):
    # This function returns ***the upper bound for the cost per example*** at test time, given the
    # costs c_fp, c_fn, and c_r.
    ngammaminus1 = int(n * cont) -1
    argsmin = (ngammaminus1, n, 1-np.exp(-T))
    argsmax = (ngammaminus1, n, np.exp(-T))
    q1 = root_scalar(Fmin, bracket=[0, 1], method='brentq', args = argsmin).root
    q2 = root_scalar(Fmin, bracket=[0, 1], method='brentq', args = argsmax).root
    return np.min([cont,q1])*c_fp + np.min([1-q2,cont])*c_fn + (q2-q1)*c_r

def Fmin(p, *args):
    # useful function
    k, n, C = args
    return binom.cdf(k, n, p) - C

def evaluate_model_with_rejection(y_pred, y_true, c_fp, c_fn, c_r):
    accept_idx = np.where(y_pred != -2)[0]
    _, _, fp, fn = confusion_matrix(y_pred[accept_idx], y_true[accept_idx])
    n_rejections = len(np.where(y_pred==-2)[0])
    cost_with_rejection = np.round((fp*c_fp + fn*c_fn + n_rejections*c_r)/len(y_true),4)
    return cost_with_rejection

def confusion_matrix(y_pred, y_true):
    # It returns the values in the confusion matrix (true positives, true negatives,
    # false positives, and false negatives).
    if np.shape(y_pred)[0] != np.shape(y_true)[0]:
        print("Predictions and true labels do not have the same size.")
        return
    fp = np.shape(np.intersect1d(np.where(y_pred == 1)[0], np.where(y_true == 0)[0]))[0]
    tp = np.shape(np.intersect1d(np.where(y_pred == 1)[0], np.where(y_true == 1)[0]))[0]
    tn = np.shape(np.intersect1d(np.where(y_pred == 0)[0], np.where(y_true == 0)[0]))[0]
    fn = np.shape(np.intersect1d(np.where(y_pred == 0)[0], np.where(y_true == 1)[0]))[0]
    return tp, tn, fp, fn


