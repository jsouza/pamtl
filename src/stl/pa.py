from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import linear_kernel
from sklearn.utils import extmath

__author__ = 'desouza'


def linear_kernel_local(X, Y, gamma=None):
    return linear_kernel(X, Y)

class PAEstimator(BaseEstimator):
    def __init__(self, loss="pai", C=0.01, n_iter=1, fit_intercept=False):
        self.loss = loss
        self.C = C
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept

        self.coef_ = None


    def _pa(self, loss_t, x_t):
        denom = extmath.norm(x_t) ** 2.0
        # special case when L_2 norm of x_t is zero (followed libol
        # implementation)
        if denom == 0:
            return 1

        d = loss_t / denom

        return d


    def _pai(self, loss_t, x_t):
        pa = self._pa(loss_t, x_t)
        return min(self.C, pa)


    def _paii(self, loss_t, x_t):
        # return loss_t / ((extmath.norm(x_t) ** 2.0) + (1.0 / (2.0 * self.C)))
        return loss_t / ((extmath.norm(x_t) ** 2.0) + (0.5 / self.C))


class KernelPAEstimator(BaseEstimator):
    def __init__(self, kernel="linear", gamma=0.01, loss="pai", C=0.01,
                 n_iter=1, fit_intercept=False):
        self.kernel = kernel
        self.gamma = gamma
        self.loss = loss
        self.C = C
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept

        self.support_ = []
        self.alphas_ = []


    def _pa(self, loss_t, kern_t):
        denom = kern_t
        # special case when L_2 norm of x_t is zero (followed libol
        # implementation)
        if denom == 0:
            return 1

        d = loss_t / denom

        return d


    def _pai(self, loss_t, x_t):
        pa = self._pa(loss_t, x_t)
        return min(self.C, pa)


    def _paii(self, loss_t, kern_t):
        # return loss_t / (kern_t + (1.0 / 2.0 * self.C))
        return loss_t / (kern_t + (0.5 * self.C))
