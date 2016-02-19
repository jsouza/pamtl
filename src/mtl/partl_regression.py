from itertools import izip

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, extmath


__author__ = 'desouza'




class PARTLRegressor(BaseEstimator, RegressorMixin):
    """
    Online Task Relationship Learning algorithm proposed by Saha et al. in
    which the task
    relationship
    matrix is dinamically learnt. Here loss is Hinge instead of Perceptron.
    """

    def __init__(self, task_num, feats_num, rounds_num=100, C=1, epsilon=0.01,
                 loss="pai", divergence="cov", eta=0.01, n_iter=1,
                 centered=True):

        self.feats_num = feats_num
        self.task_num = task_num
        if self.task_num < 1:
            raise ValueError("number of tasks must be greater than 1.")


        # initialize interaction matrix with independent learners
        self.A = 1.0 / task_num * np.identity(task_num)

        # hyper-parameters
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.divergence = divergence
        self.n_iter = n_iter
        self.eta = eta
        # number of rounds of priming A
        self.rounds_num = rounds_num

        self.centered = centered

        # initialize w's with d x K positions (d = feats, K = tasks)
        self.coef_ = np.zeros(self.feats_num * self.task_num)
        # averaged model
        self.avg_coef_ = np.copy(self.coef_)

        # number of instances seen
        self.t = 0
        # number of instances discarded (zero-arrays)
        self.discarded = 0

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
        if self.C < pa:
            return self.C

        return pa

    def _paii(self, loss_t, x_t):
        return loss_t / (extmath.norm(x_t) ** 2.0) + 1.0 / 2.0 * self.C


    def _get_task_id(self, X_inst, feats_num):
        a = np.nonzero(X_inst != 0)
        first_element = a[0][0]
        task_num = first_element / feats_num
        return task_num

    def _sym(self, X):
        temp = (X + X.T) / 2.0
        return temp

    def _batch_opt(self, W):
        num = np.dot(W.T, W) ** (1.0 / 2.0)
        denom = np.trace(num)
        bo = num / denom
        return bo

    def _log_det_div(self, W):
        prev_a_inv = np.linalg.inv(self.A)
        # prev_a_inv = np.linalg.pinv(self.A)
        WT = W.T
        dot_w = np.dot(WT, W)
        symdot = self._sym(dot_w)
        log_det = prev_a_inv + self.eta * symdot
        # return log_det

        log_det_inv = np.linalg.inv(log_det)
        # log_det_inv = np.linalg.pinv(log_det)
        return log_det_inv

    def _von_neumann_div(self, W):
        dot_w = np.dot(W.T, W)
        # log_a = np.log(self.A) - self.eta * self._sym(dot_w)
        log_a = sp.linalg.logm(self.A) - self.eta * self._sym(dot_w)
        # exp_log_a = np.exp(log_a)
        exp_log_a = sp.linalg.expm(log_a)

        return exp_log_a


    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y)

        for x_i, y_i in izip(X, y):
            self.partial_fit(x_i.reshape(-1, 1), y_i.reshape(1, -1))

        return self


    def partial_fit(self, X_t, y_t):
        # if all features are zero, discard example
        if np.sum(X_t) == 0:
            self.discarded += 1
            return self

        # updates the number of instances seen
        self.t += 1

        reg_func = self._pai
        if self.loss == "pa":
            reg_func = self._pa
        elif self.loss == "pai":
            reg_func = self._pai
        elif self.loss == "paii":
            reg_func = self._paii

        for _ in xrange(self.n_iter):
            # gets prediction based on current model
            y_hat_t = np.dot(self.coef_, X_t.T)

            # calculates difference between prediction and actual value
            discrepancy_t = np.abs(y_hat_t - y_t)
            #print discrepancy_t.shape, discrepancy_t


            # wx_dot = np.dot(self.coef_, X_t)
            # y_hat_t = np.sign(wx_dot)
            # loss_t = max([0, (1 - y_t * wx_dot)])

            # computes epsilon-hinge loss
            loss_t = 0
            if discrepancy_t > self.epsilon:
                loss_t = discrepancy_t - self.epsilon

            tau_t = reg_func(loss_t, X_t)

            task_id = self._get_task_id(X_t, self.feats_num)

            for task in xrange(self.task_num):
                # for indexing task weights that change in the for loop
                begin = task * self.feats_num
                end = begin + self.feats_num

                # for indexing the task of X_t
                tbegin = task_id * self.feats_num
                tend = tbegin + self.feats_num

                # computes new coefs
                new_coef = np.sign(y_t - y_hat_t) * self.A[
                    task, task_id] * tau_t * X_t[tbegin:tend]

                # updates coefs
                self.coef_[begin:end] += new_coef

            self.avg_coef_ += self.coef_

            # updates A
            if self.t >= self.rounds_num:
                # first, reshape coefs (w in the paper) to W
                # which is the d x K matrix where d are the
                # features and K the different tasks
                w = np.copy(self.coef_)
                W = w.reshape((self.task_num, self.feats_num)).T

                # update interaction matrix
                if self.divergence == "cov":
                    covA = np.cov(W, rowvar=0)
                    if self.centered:
                        self.A = covA
                    else:
                        self.A = np.linalg.inv(covA)

                elif self.divergence == "corrcoef":
                    corrcoefW = np.corrcoef(W, rowvar=0)
                    if self.centered:
                        self.A = corrcoefW
                    else:
                        self.A = np.linalg.inv(corrcoefW)

                elif self.divergence == "vn":
                    self.A = self._von_neumann_div(W)

                elif self.divergence == "ld":
                    self.A = self._log_det_div(W)

                elif self.divergence == "bo":
                    self.A = self._batch_opt(W)
                else:
                    raise ValueError("divergence mode not valid")

                    # np.fill_diagonal(self.A, 1)

        return self

    def predict(self, X, averaged=False):
        X = check_array(X)

        # if self.fit_intercept:
        # X = np.column_stack((X, np.ones(X.shape[0])))

        # print self.coef_.shape
        # print X.shape
        if not averaged:
            y_preds = np.dot(self.coef_, X.T)
        else:
            y_preds = np.dot(self.avg_coef_, X.T)

        return y_preds



