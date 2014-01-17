
import numpy as np
import os

N = 10000
f_a, f_b = (1.0, 0.0)
g_a, g_b = (1.0, 1.0)

#assert(f_a < 1.0)
#assert(g_a < 1.0)

burnin = 1000

output_path = "/u/alaingui/umontreal/gsn_exploration/code/01_twin_exponentials"
#plot_filename = os.path.join(path, "plot_N%d_%0.2f_%0.2f_%0.2f_%0.2f.png" % (N, f_a, f_b, g_a, g_b))
plot_X_filename = os.path.join(output_path, "X_02_N%d_%0.2f_%0.2f_%0.2f_%0.2f.png"    % (N, f_a, f_b, g_a, g_b))
plot_H_filename = os.path.join(output_path, "H_02_N%d_%0.2f_%0.2f_%0.2f_%0.2f.png"    % (N, f_a, f_b, g_a, g_b))
data_filename   = os.path.join(output_path, "data_02_N%d_%0.2f_%0.2f_%0.2f_%0.2f.pkl" % (N, f_a, f_b, g_a, g_b))

def one_step(H,X):
    # X_t from H_t
    # H_{t+1} from X_t
    # X = g_b + X * np.random.exponential(scale=g_a, size=H.shape)
    # H = f_b + X * np.random.exponential(scale=f_a, size=X.shape)
    X = (1.0 - 2.0 * (np.random.rand(H.shape[0]) > 0.5)) * np.random.exponential(scale=(g_b + g_a*np.sqrt(np.abs(H))), size=H.shape)
    H = f_b + np.sqrt(f_a * np.abs(X)) * np.random.normal(1.0, size=X.shape)
    return (H,X)

X = np.ones((N,))
H = np.ones((N,))

for _ in range(burnin):
    (H,X) = one_step(H,X)


def plot_dataset(H, X, plot_X_filename, plot_H_filename):

    import matplotlib
    # This has already been specified in .scitools.cfg
    # so we don't need to explicitly pick 'Agg'.
    # matplotlib.use('Agg')
    import pylab
    import matplotlib.pyplot as plt

    nbins = 100
    dpi = 150

    pylab.hist(H, bins=100, color='#f9a21d')
    pylab.draw()
    pylab.savefig(plot_H_filename, dpi=dpi)
    pylab.close()
    print "Wrote %s." % (plot_H_filename,)

    pylab.hist(X, bins=100, color='#687ff4')
    pylab.draw()
    pylab.savefig(plot_X_filename, dpi=dpi)
    pylab.close()
    print "Wrote %s." % (plot_X_filename,)


print "X ranges from %f to %f" % (X.min(), X.max())
print "H ranges from %f to %f" % (H.min(), H.max())

plot_dataset(H, X, plot_X_filename, plot_H_filename)

import cPickle
cPickle.dump({'X':X, 'H':H}, open(data_filename, "w"))
print "Wrote %s." % data_filename
