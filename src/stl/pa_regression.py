from itertools import izip

from sklearn.base import RegressorMixin
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_regression
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import rbf_kernel, \
    polynomial_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
import numpy as np

from pa import PAEstimator, KernelPAEstimator, linear_kernel_local


__author__ = 'desouza'


class PARegressor(PAEstimator, RegressorMixin):
    def __init__(self, C=1, epsilon=0.01, loss="pai", n_iter=1,
                 fit_intercept=False):
        super(PARegressor, self).__init__(loss, C, n_iter, fit_intercept)

        self.epsilon = epsilon


    def partial_fit(self, X, y):
        # check_arrays(X, y)

        if self.loss == "pa":
            reg_func = self._pa
        elif self.loss == "pai":
            reg_func = self._pai
        elif self.loss == "paii":
            reg_func = self._paii

        # inits coefficients
        if self.coef_ is None:
            self.coef_ = np.zeros(X.shape[0])

        for _ in xrange(self.n_iter):
            y_hat_t = np.dot(self.coef_, X)
            discrepancy_t = np.abs(y - y_hat_t)
            loss_t = 0
            if discrepancy_t > self.epsilon:
                loss_t = discrepancy_t - self.epsilon

            tau_t = reg_func(loss_t, X)

            self.coef_ = self.coef_ + (np.sign(y - y_hat_t) * tau_t * X)

        return self


    def fit(self, X, y):
        check_array(X, y)

        for x_i, y_i in izip(X, y):
            self.partial_fit(x_i, y_i)

        return self


    def predict(self, X):
        X = check_array(X)

        # print self.coef_.shape
        # print X.shape
        y_preds = np.dot(self.coef_, X.T)
        return y_preds


class KernelPARegressor(KernelPAEstimator, RegressorMixin):
    def __init__(self, kernel="linear", gamma=0.01, C=1, epsilon=0.01,
                 loss="pa", n_iter=1,
                 fit_intercept=False):
        super(KernelPARegressor, self).__init__(kernel, gamma, loss, C, n_iter,
                                                fit_intercept)

        self.epsilon = epsilon


    def predict(self, X):
        X = check_array(X)

        kern_func = linear_kernel_local
        if self.kernel == "rbf":
            kern_func = rbf_kernel
        elif self.kernel == "poly":
            kern_func = polynomial_kernel

        support = np.array(self.support_)
        alphas = np.array(self.alphas_)

        phi = kern_func(support, X, self.gamma)
        margins = np.dot(phi.T, alphas)

        y_preds = margins

        return y_preds

    def fit(self, X, y):
        check_array(X, y)

        for x_i, y_i in izip(X, y):
            self.partial_fit(x_i, y_i)

        return self


    def partial_fit(self, X, y):

        reg_func = self._pai
        if self.loss == "pa":
            reg_func = self._pa
        elif self.loss == "paii":
            reg_func = self._paii

        kern_func = linear_kernel_local
        if self.kernel == "rbf":
            kern_func = rbf_kernel
        elif self.kernel == "poly":
            kern_func = polynomial_kernel

        # computes kernel and prediction
        if len(self.support_) > 0:
            # kernel function (phi) value
            # computes phi of support vectors against current example t
            support_t = np.array(self.support_)
            phi_t = kern_func(support_t, X, self.gamma)
            # takes the prediction
            alphas = np.array(self.alphas_)
            y_hat_t = np.dot(alphas.T, phi_t.ravel())

        else:  # support has 0 elements
            # then prediction is zero
            y_hat_t = 0

        # computes loss
        discrepancy_t = np.abs(y - y_hat_t)
        loss_t = 0
        if discrepancy_t > self.epsilon:
            loss_t = discrepancy_t - self.epsilon
        # loss_t = max(0, np.abs(y - y_hat_t) - self.epsilon)
        # loss_t = max(0, np.abs(y_hat_t - y) - self.epsilon)

            # computes weight for new support vector and updates the model
            kern_t = np.ravel(kern_func(X, X, self.gamma))
            # print kern_t

            tau_t = reg_func(loss_t, kern_t)
            alpha_t = np.sign(y - y_hat_t) * tau_t

            if alpha_t != 0:
                self.support_.append(X)
                self.alphas_.append(alpha_t)

        return self


def main_qeda_1000():
    # boston = load_boston()
    # X = boston.data
    # y = boston.target
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    # test_size=0.3, random_state=40)

    X_train = np.loadtxt(
        '/home/desouza/Projects/qe_da/domains_large_bb17/sample-nomarkup-1000'
        '/stl_40/features/1.0-it.src.csv',
        delimiter=',')
    y_train = np.loadtxt(
        '/home/desouza/Projects/qe_da/domains_large_bb17/sample-nomarkup-1000'
        '/stl_40/labels/1.0-it.src.hter')

    X_test = np.loadtxt(
        '/home/desouza/Projects/qe_da/domains_large_bb17/sample-nomarkup-1000'
        '/stl_40/features/it.tgt.csv',
        delimiter=',')
    y_test = np.loadtxt(
        '/home/desouza/Projects/qe_da/domains_large_bb17/sample-nomarkup-1000'
        '/stl_40/labels/it.tgt.hter')

    # X_train_bias = np.column_stack(tuple([X_train, np.ones(X_train.shape[
    # 0])]))
    # X_test_bias = np.column_stack(tuple([X_test, np.ones(X_test.shape[0])]))


    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "epsilon": [0.0001, 0.001, 0.01, 0.1],
        "loss": ["pa", "pai", "paii"]}

    search = GridSearchCV(PARegressor(), param_grid,
        scoring='mean_absolute_error', n_jobs=8, iid=True,
        refit=True, cv=5, verbose=1)

    search.fit(X_train, y_train)
    # est = PARegressor(C=0.4, epsilon=0.01, loss="paii")

    # for i in range(10):
    # for x_i, y_i in izip(X_train_bias, y_train):
    # for x_i, y_i in izip(X_train, y_train):
    # est.partial_fit(x_i, y_i)
    # est.fit(X_train, y_train)


    # y_preds = est.predict(X_test_bias)
    # y_preds = est.predict(X_test)
    y_preds = search.predict(X_test)

    # print est.coef_
    print search.best_estimator_.coef_
    print search.best_params_

    print y_preds.shape

    mae_my_pa = mean_absolute_error(y_preds, y_test)
    print "My PA MAE = %2.4f" % mae_my_pa

    param_grid = {
        "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "epsilon": [0.0001, 0.001, 0.01, 0.1],
        "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"]}

    search = GridSearchCV(PassiveAggressiveRegressor(), param_grid,
        scoring='mean_absolute_error', n_jobs=8, iid=True,
        refit=True, cv=5, verbose=1)
    # est_pa = PassiveAggressiveRegressor()
    # est_pa.partial_fit(X_train, y_train)
    search.fit(X_train, y_train)
    # y_preds_pa = est_pa.predict(X_test)
    y_preds_pa = search.predict(X_test)
    mae_sk_pa = mean_absolute_error(y_preds_pa, y_test)
    print "Sklearn PA MAE = %2.4f" % mae_sk_pa

    # print est_pa.coef_
    print search.best_estimator_.coef_
    print search.best_params_


def main():
    X, y, coef = make_regression(1000, 200, 10, 1, noise=0.05, coef=True,
                                 random_state=42)

    # X = np.column_stack((X, np.ones(X.shape[0])))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)

    # sca = StandardScaler()
    # sca.fit(X_train)
    # X_train = sca.transform(X_train)
    # X_test = sca.transform(X_test)

    # print X.shape
    # print y.shape
    # print coef.shape

    param_grid = {
        "C": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10,
              100, 1000],
        "epsilon": [0.0001, 0.001, 0.01, 0.1]}

    param_grid_kern = {
        "C": [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10,
              100, 1000],
        "epsilon": [0.0001, 0.001, 0.01, 0.1],
        "gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
    # "loss": ["pa", "pai", "paii"]}}

    my_pa = PARegressor(loss="paii", C=1, epsilon=0.001, n_iter=1,
                        fit_intercept=False)
    #
    # search = GridSearchCV(my_pa, param_grid,
    #                       scoring='mean_absolute_error', n_jobs=8, iid=True, refit=True, cv=5,
    #                       verbose=1)
    # search.fit(X_train, y_train)
    # print search.best_params_

    my_pa.fit(X_train, y_train)
    print my_pa.coef_

    # y_preds = search.predict(X_test)
    y_preds = my_pa.predict(X_test)

    mae_my_pa = mean_absolute_error(y_test, y_preds)
    print "My PA MAE = %2.4f" % mae_my_pa

    my_kpa_linear = KernelPARegressor(kernel="linear", loss="paii", C=1, epsilon=0.001, n_iter=1, fit_intercept=False)
    my_kpa_linear.fit(X_train, y_train)
    print "alphas", len(my_kpa_linear.alphas_), my_kpa_linear.alphas_
    y_preds = my_kpa_linear.predict(X_test)
    mae_kpa_linear = mean_absolute_error(y_test, y_preds)
    print "My KPA linear MAE = %2.4f" % mae_kpa_linear

    my_kpa_rbf = KernelPARegressor(kernel="rbf", loss="paii", gamma=0.001, C=1, epsilon=0.001, n_iter=1, fit_intercept=False)
    # search = GridSearchCV(my_kpa_rbf, param_grid_kern,
    #                       scoring='mean_absolute_error', n_jobs=8, iid=True, refit=True, cv=5,
    #                       verbose=1)
    # search.fit(X_train, y_train)

    my_kpa_rbf.fit(X_train, y_train)
    print "alphas", len(my_kpa_rbf.alphas_), my_kpa_rbf.alphas_
    print "support", len(my_kpa_rbf.support_)
    # print "alphas", len(search.best_estimator_.alphas_)  # , my_kpa_rbf.alphas_
    # print "support", len(search.best_estimator_.support_)
    # print search.best_params_
    y_preds = my_kpa_rbf.predict(X_test)
    # y_preds = search.predict(X_test)
    mae_my_kpa = mean_absolute_error(y_test, y_preds)
    print "My Kernel PA MAE = %2.4f" % mae_my_kpa

    # print search.best_estimator_
    # print np.corrcoef(search.best_estimator_.coef_, coef)

    # param_grid = {
    # "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 10,
    #           100, 1000, 10000],
    #     "epsilon": [0.0001, 0.001, 0.01, 0.1],
    #     # "loss": ["epsilon_insensitive", "squared_epsilon_insensitive"]}
    #     "loss": ["squared_epsilon_insensitive"]}


    # search = GridSearchCV(PassiveAggressiveRegressor(fit_intercept=True),
    # param_grid, scoring='mean_absolute_error', n_jobs=8, iid=True,
    # refit=True, cv=5, verbose=1)
    # search.fit(X_train, y_train)

    sk_pa = PassiveAggressiveRegressor(loss="squared_epsilon_insensitive", C=1,
                                       epsilon=0.001, n_iter=1,
                                       fit_intercept=False,
                                       warm_start=True)
    for i in xrange(X_train.shape[0]):
        # for x_i, y_i in zip(X_train, y_train):
        x = np.array(X_train[i], ndmin=2)
        y = np.array(y_train[i], ndmin=1)
        # print x.shape
        # print y
        sk_pa.partial_fit(x, y)

    # sk_pa.fit(X_train, y_train)

    # y_preds = search.predict(X_test)
    y_preds = sk_pa.predict(X_test)
    mae_sk_pa = mean_absolute_error(y_preds, y_test)
    print "Sklearn PA MAE = %2.4f" % mae_sk_pa
    # print search.best_estimator_
    # print np.corrcoef(search.best_estimator_.coef_, coef)

    # en = ElasticNet(fit_intercept=True)
    # en.fit(X_train, y_train)
    # y_preds = en.predict(X_test)
    # mae_en = mean_absolute_error(y_test, y_preds)
    # print "Sklearn EN MAE = %2.4f" % mae_en

    # svr = SVR(kernel="linear")
    # svr.fit(X_train, y_train)
    # y_preds = svr.predict(X_test)
    # mae_svr = mean_absolute_error(y_test, y_preds)
    # print "Sklearn SVR MAE = %2.4f" % mae_svr

    # sgd = SGDRegressor(loss="epsilon_insensitive", penalty="l2",
    # epsilon=0.001)
    # # sgd.fit(X_train, y_train)
    # for i in xrange(X_train.shape[0]):
    # # for x_i, y_i in zip(X_train, y_train):
    # x = np.array(X_train[i], ndmin=2)
    #     y = np.array(y_train[i], ndmin=1)
    #     # print x.shape
    #     # print y
    #     sgd.partial_fit(x, y)
    #
    # y_preds = sgd.predict(X_test)
    # mae_sgd = mean_absolute_error(y_test, y_preds)
    # print "Sklearn SGD MAE = %2.4f" % mae_sgd


if __name__ == "__main__":
    main()
    # main_qeda_1000()

