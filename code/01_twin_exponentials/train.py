
import numpy as np
import os

N = 10000
f_true_a, f_true_b = (1.0, 0.0)
g_true_a, g_true_b = (1.0, 1.0)

output_path = "/u/alaingui/umontreal/gsn_exploration/code/01_twin_exponentials"
data_filename   = os.path.join(output_path, "data_02_N%d_%0.2f_%0.2f_%0.2f_%0.2f.pkl" % (N,
                                                                                         f_true_a, f_true_b,
                                                                                         g_true_a, g_true_b))

import cPickle
data = cPickle.load(open(data_filename, "r"))
#X = data["X"]

# put a function here that samples H from X

# put a function here that samples X from H

import theano
import theano.tensor as T

f_params = (theano.shared(0.8, 'f_a'),
            theano.shared(0.8, 'f_b'))

g_params = (theano.shared(0.8, 'g_a'),
            theano.shared(0.8, 'g_b'))

def f(X, H):
    return T.exp(log_f(X, H))

def g(X, H):
    return T.exp(log_g(X, H))

def log_f(X, H):
    # H given X
    a = f_params[0]
    b = f_params[1]
    return -0.5 * T.log(2 * np.pi * abs(X)) - 0.5 * (H - b)**2 / abs(X)

def log_g(X, H):
    # X given H
    a = g_params[0]
    b = g_params[1]
    s = b + a * T.sqrt(abs(H))
    return - T.log(s * 2.0) - abs(X) / s

X = T.vector()
H = T.vector()
func_log_f = theano.function([X, H], log_f(X, H))
func_log_g = theano.function([X, H], log_g(X, H))

print func_log_f(np.array([1.0]),
                 np.array([1.0]))

