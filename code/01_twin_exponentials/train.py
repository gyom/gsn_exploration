
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
# cut the size for testing
data['X'] = data['X'][0:1000]

# put a function here that samples H from X

# put a function here that samples X from H

import theano
import theano.tensor as T

import numpy as np

want_true_values = False
if want_true_values:
    f_params = (theano.shared(f_true_a, 'f_a'),
                theano.shared(f_true_b, 'f_b'))

    g_params = (theano.shared(g_true_a, 'g_a'),
                theano.shared(g_true_b, 'g_b'))
else:
    f_params = (theano.shared(1.0, 'f_a'),
                theano.shared(0.0, 'f_b'))

    g_params = (theano.shared(0.7, 'g_a'),
                theano.shared(1.3, 'g_b'))


def f(X, H):
    return T.exp(log_f(X, H))

def g(X, H):
    return T.exp(log_g(X, H))

def log_f(X, H):
    # H given X
    a = f_params[0]
    b = f_params[1]
    return -0.5 * T.log(2 * np.pi * a * abs(X)) - 0.5 * (H - b)**2 / (a * abs(X))

def log_g(X, H):
    # X given H
    a = g_params[0]
    b = g_params[1]
    s = b + a * T.sqrt(abs(H))
    return - T.log(s * 2.0) - abs(X) / s


def sample_H_given_X(X):
    a = f_params[0].get_value()
    b = f_params[1].get_value()
    H = b + np.sqrt(a * np.abs(X)) * np.random.normal(1.0, size=X.shape)
    return H

def sample_X_given_H(H):
    a = g_params[0].get_value()
    b = g_params[1].get_value()
    X = (1.0 - 2.0 * (np.random.rand(H.shape[0]) > 0.5)) * np.random.exponential(scale=(b + a*np.sqrt(np.abs(H))), size=H.shape)    
    return X


thX = T.vector()
thH = T.vector()
func_log_f = theano.function([thX, thH], log_f(thX, thH))
func_log_g = theano.function([thX, thH], log_g(thX, thH))

func_softmax_log_g   = theano.function([thX, thH], T.nnet.softmax(log_g(thX, thH)))
func_softmax_log_g_g = theano.function([thX, thH], [T.nnet.softmax(log_g(thX, thH)), g(thX, thH)])


th_log_g = log_g(thX, thH)
th_g = T.exp(th_log_g)
th_log_f = log_f(thX, thH)
th_f = T.exp(th_log_f)

th_J_log_g = theano.gradient.jacobian(th_log_g, wrt=g_params)
th_J_log_f = theano.gradient.jacobian(th_log_f, wrt=f_params)

func_compute_all = theano.function([thX, thH], [th_log_g, th_g,
                                                th_J_log_g[0], th_J_log_g[1],
                                                th_log_f, th_f,
                                                th_J_log_f[0], th_J_log_f[1]])
#func_compute_all = theano.function([thX, thH], [th_log_g, th_g, th_log_f, th_f])


# It would be possible to rewrite this with
# X and H as 2D meshes, but that would still
# involve the assumption that both were
# originally 1D.
def compute_omega_qX(X, H = None):
    # X comes from the training set
    # and NOT from sampled values
    # with your own parameters.
    # However, you will sample H
    # that do rely on your
    # current parameters.

    # The rows of omega will sum to 1.0 .

    N = X.shape[0]
    omega = np.zeros((N,N))
    qX_precursor = np.zeros((N,N))
    qX = np.zeros((N,))

    # sample H particles for
    # each original x \in X
    if H is None:
        H = sample_H_given_X(X)

    for (i,x) in enumerate(X):
        xtile = x * np.ones((N,))
        #omega[i, :] = func_softmax_log_g(xtile, H)
        #qX[i] = omega[i,:].mean()
        (omega[i, :], qX_precursor[i, :]) = func_softmax_log_g_g(xtile, H)
        qX[i] = qX_precursor[i,:].sum()

    return (omega, qX)

#(omega, qX) = compute_omega_qX(data['X'])
#print "------ omega ------"
#print omega
#print "-------------------\n"
#print "mean log q(X) is %f" % np.log(qX).mean()

def one_iteration_SVD(X, learning_rate = 1.0):

    H = sample_H_given_X(X)

    # could be optimized better, but we're just losing a
    # factor of two in the worst case
    omega, qX = compute_omega_qX(X, H)
    print "mean log q(X) is %f" % np.log(qX).mean()

    _, _, J_log_ga, J_log_gb, _, _, J_log_fa, J_log_fb = func_compute_all(X, H)

    # that mean(axis=0) could be a sum, but then
    # we'd have the gradient of the loss depend
    # on the number of training points
    subs_J_log_g = (omega.dot(J_log_ga).mean(axis=0), omega.dot(J_log_gb).mean(axis=0))
    subs_J_log_f = (omega.dot(J_log_fa).mean(axis=0), omega.dot(J_log_fb).mean(axis=0))

    print "\tsubs_J_log_g : %s" % str(subs_J_log_g)
    print "\tsubs_J_log_f : %s" % str(subs_J_log_f)

    # then do the updates
    for k in [0,1]:
        g_params[k].set_value(g_params[k].get_value() + learning_rate * subs_J_log_g[k])
        #f_params[k].set_value(f_params[k].get_value() + learning_rate * subs_J_log_f[k])

    print "\tf_params = (%0.4f, %0.4f), g_params = (%0.4f, %0.4f)" % (f_params[0].get_value(),
                                                                      f_params[1].get_value(),
                                                                      g_params[0].get_value(),
                                                                      g_params[1].get_value())

#for iter in range(1000):
#    one_iteration_SVD(data['X'], 0.1)


def plot_gradient_trajectory_g_params(X, plot_filename):

    import matplotlib
    # This has already been specified in .scitools.cfg
    # so we don't need to explicitly pick 'Agg'.
    # matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    N_a, N_b = 10, 10

    A, B = np.meshgrid(np.linspace(0.5, 2.0, N_a),
                       np.linspace(0.5, 2.0, N_b))

    Z = np.zeros(B.shape)
    # values Z[n_b, n_a] will be set
    # through two for loops
    grad_ZA = np.zeros(A.shape)
    grad_ZB = np.zeros(B.shape)

    for n_a in range(N_a):
        a = A[0, n_a]
        g_params[0].set_value(a)

        for n_b in range(N_b):
            b = B[n_b, 0]
            g_params[1].set_value(b)

            H = sample_H_given_X(X)

            # could be optimized better, but we're just losing a
            # factor of two in the worst case
            omega, qX = compute_omega_qX(X, H)
            #print "mean log q(X) is %f" % np.log(qX).mean()
            Z[n_b, n_a] = np.log(qX).mean()

            _, _, J_log_ga, J_log_gb, _, _, J_log_fa, J_log_fb = func_compute_all(X, H)
            subs_J_log_g = (omega.dot(J_log_ga).mean(axis=0), omega.dot(J_log_gb).mean(axis=0))
            #subs_J_log_f = (omega.dot(J_log_fa).mean(axis=0), omega.dot(J_log_fb).mean(axis=0))

            grad_ZA[n_a, n_b] = subs_J_log_g[0]
            grad_ZB[n_a, n_b] = subs_J_log_g[1]

        print "Done with a=%f." % a


    dpi=150
    pylab.hold(True)
    plt.contourf(A, B, Z)
    pylab.quiver(A, B, grad_ZA, grad_ZB)
    pylab.draw()
    pylab.savefig(plot_filename, dpi=dpi)
    pylab.close()
    print "Wrote %s." % (plot_filename,)


plot_filename = os.path.join(output_path,
                             "contourscatter_02_N%d_%0.2f_%0.2f_%0.2f_%0.2f.png" % (N,
                                                                                    f_true_a, f_true_b,
                                                                                    g_true_a, g_true_b))
plot_gradient_trajectory_g_params(data['X'], plot_filename)



