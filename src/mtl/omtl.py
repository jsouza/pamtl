import glob
from itertools import izip
import os
import sys

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, extmath


__author__ = 'desouza'


def compound_descriptor_online(features, task_id, task_num, feats_num):
    num_samples, num_feats = features.shape
    if num_feats != feats_num:
        raise ValueError(
            "number of features is different than the one declared.")

    base = task_id * num_feats
    offset = base + num_feats
    task_block = np.zeros((num_samples, num_feats * task_num))
    task_block[:, base:offset] = features

    return task_block


def compound_descriptor(task_features, task_id, task_num):
    task_features = check_array(task_features)

    num_samples, num_feats = task_features.shape

    base = task_id * num_feats
    offset = base + num_feats

    task_block = np.zeros((num_samples, num_feats * task_num))
    task_block[:, base:offset] = task_features
    return task_block


def compound_descriptor_at_once(tasks_features, tasks_labels):
    """
    Prepares the input for the MTL Perceptron.

    :param tasks_features list containing the feature vectors of the tasks (
    each task is one item in the list)
    :param tasks_labels list containing the labels of the vectors for each task
    """

    task_num_feats = len(tasks_features)
    task_num_labels = len(tasks_labels)

    if task_num_feats != task_num_labels:
        raise ValueError("number of tasks differ for features and labels.")

    # checks if all tasks have the same number of features
    num_feats = 0
    for i, task_feats in enumerate(tasks_features):
        num_feats = task_feats.shape[1]
        if i > 0:
            if task_feats.shape[1] != num_feats:
                raise ValueError(
                    "number of features is different among the tasks.")

    tasks_labels = [np.atleast_2d(labels).T for labels in tasks_labels]

    compound_feats = []
    for task_id, task_feats in enumerate(tasks_features):
        base = task_id * num_feats
        offset = base + num_feats
        num_samples, num_feats = task_feats.shape
        task_block = np.zeros((num_samples, num_feats * task_num_feats))
        task_block[:, base:offset] = task_feats
        compound_feats.append(task_block)

    feats_arr = np.row_stack(tuple(compound_feats))
    labels_arr = np.row_stack(tuple(tasks_labels))

    return feats_arr, labels_arr


class OMTLClassifier(BaseEstimator, ClassifierMixin):
    """
    Linear Perceptron Classifier proposed by Cavallanti et al.
    """

    def __init__(self, task_num, feats_num, interaction="half"):
        self.task_num = task_num
        self.feats_num = feats_num

        # inits coeficients to the number of tasks * features
        self.coef_ = np.zeros(self.feats_num * self.task_num)

        # inits interaction matrix
        # by default, half update
        self.A = (1.0 / (task_num + 1)) * ((np.identity(task_num) + 1) * 2)
        if interaction == "ipl":
            self.A = np.identity(self.task_num)
            # self.A = (1.0 / task_num) * np.identity(self.task_num)

        # self.A = np.linalg.inv(self.A)

        # number of instances seen
        self.t = 0
        # number of instances discarded for being all zeroes
        self.discarded = 0
        # number of updates made to the coeficients
        self.s = 0

    def _get_task_id(self, X_inst):
        a = np.nonzero(X_inst != 0)
        first_element = a[0][0]
        task_num = first_element / self.feats_num
        return task_num


    def fit(self, X, y):
        # y = np.atleast_2d(y).T

        X, y = check_array(X, y)

        for x_i, y_i in izip(X, y):
            self.partial_fit(x_i, y_i)

        return self


    def partial_fit(self, X_t, y_t):
        # checks if features are non zero
        if np.sum(X_t) == 0:
            self.discarded += 1
            return self

        # updates the number of instances seen
        self.t += 1

        task_id_t = self._get_task_id(X_t)

        y_pred_t = np.sign(np.dot(self.coef_, X_t))
        if y_pred_t != y_t:
            kron = np.dot(np.kron(self.A, np.identity(self.feats_num)), X_t)
            # print kron.shape
            self.coef_ = self.coef_ + y_t * kron
            self.s += 1


    # def partial_fit(self, X_t, y_t):
    # # checks if features are non zero
    # if np.sum(X_t) == 0:
    # self.discarded += 1
    #         return self
    #
    #     # updates the number of instances seen
    #     self.t += 1
    #
    #     task_id_t = self._get_task_id(X_t)
    #
    #     y_pred_t = np.sign(np.dot(self.coef_, X_t))
    #     if y_pred_t != y_t:
    #         tbegin = task_id_t * self.feats_num
    #         tend = tbegin + self.feats_num
    #
    #         for task in xrange(self.task_num):
    #             begin = task * self.feats_num
    #             end = begin + self.feats_num
    #             self.coef_[begin:end] = self.coef_[begin:end] + y_t *
    # self.A[task, task_id_t] * X_t[tbegin:tend]
    #
    #         self.s += 1

    def predict(self, X):
        X = check_array(X)

        y_preds = np.sign(np.dot(self.coef_, X.T))
        return y_preds




def main_mtl():
    input_dir = "/home/desouza/Projects/qe_da/domains_large_bb17/"

    domains_names = ["it", "ted", "wsd"]
    tasks_features = []
    tasks_labels = []
    est = PAMTLRegressor(3, 17)

    for task_id, domain_name in enumerate(domains_names):
        # tasks_features.append(
        # np.loadtxt(glob.glob(input_dir + os.sep + "features/" + domain_name
        # + "*.tsv")[0]))
        X_train = np.loadtxt(
            glob.glob(input_dir + os.sep + "features/" + domain_name + "*.tsv")[
                0])
        X_train[np.isnan(X_train)] = 0
        task_block = compound_descriptor(X_train, task_id, len(domains_names))

        # tasks_labels.append(
        # np.loadtxt(glob.glob(input_dir + os.sep + "labels/" +
        # domain_name + "*.hter")[0], ndmin=2))
        y_train = np.loadtxt(
            glob.glob(input_dir + os.sep + "labels/" + domain_name + "*.hter")[
                0], ndmin=2)

        print task_block.shape
        print y_train.shape

        est.fit(task_block, y_train)

    print est.coef_


def main_mtl_test():
    shuffles = int(sys.argv[1])
    work_dir = sys.argv[2]
    domains = ["it", "wsd", "ted"]
    for seed in range(40, 40 + shuffles, 1):
        print "### ", seed
        input_dir = work_dir + os.sep + "stl_" + str(seed) + os.sep

        src_feats_list = []
        src_labels_list = []
        for task_id, src_domain in enumerate(domains):
            src_feat_paths = glob.glob(
                input_dir + "features/*" + src_domain + "*src.csv")
            src_label_paths = glob.glob(
                input_dir + "labels/*" + src_domain + "*src.hter")

            if len(src_feat_paths) != len(src_label_paths):
                print "number of source feature files and label files " \
                      "differs: %d " \
                      "and %d" % (
                          len(src_feat_paths), len(src_label_paths))
                sys.exit(1)

            src_feat_paths_sorted = sorted(src_feat_paths)
            src_label_paths_sorted = sorted(src_label_paths)

            # for i in range(len(src_feat_paths_sorted)):
            # only 100%
            for i in [9]:
                X_src = np.nan_to_num(
                    np.loadtxt(src_feat_paths_sorted[i], delimiter=","))

                y_src = np.clip(np.loadtxt(src_label_paths_sorted[i], ndmin=2),
                                0, 1)

                X_src_comp = compound_descriptor(X_src, task_id, len(domains))

                # print X_src_comp.shape

            src_feats_list.append(X_src_comp)
            src_labels_list.append(y_src)

        X_multiple_src = np.row_stack(tuple(src_feats_list))
        y_multiple_src = np.row_stack(tuple(src_labels_list))

        print X_multiple_src.shape
        print y_multiple_src.shape

        # shuffled_rows = np.arange(X_multiple_src.shape[0])
        # np.random.shuffle(shuffled_rows)

        # X_src_shuffled = X_multiple_src[shuffled_rows,:]
        # y_src_shuffled = y_multiple_src[shuffled_rows,:]

        # print X_src_shuffled.shape
        # print y_src_shuffled.shape



        param_grid = {
            "C": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1.0],
            "epsilon": [0.0001, 0.001, 0.01, 0.1],
            "loss": ["pa", "pai", "paii"]}

        search = GridSearchCV(PAMTLRegressor(3, 17), param_grid,
                              scoring='mean_absolute_error', n_jobs=8,
                              iid=False, refit=True, cv=5, verbose=1)


        # est.fit(X_src_shuffled, y_src_shuffled)
        # est.fit(X_multiple_src, y_multiple_src)
        search.fit(X_multiple_src, y_multiple_src)

        for task_id, tgt_domain in enumerate(domains):
            tgt_feat_paths = glob.glob(
                input_dir + "features/*" + tgt_domain + "*tgt.csv")
            tgt_label_paths = glob.glob(
                input_dir + "labels/*" + tgt_domain + "*tgt.hter")

            tgt_feat_paths_sorted = sorted(tgt_feat_paths)
            tgt_label_paths_sorted = sorted(tgt_label_paths)

            X_tgt = np.nan_to_num(
                np.loadtxt(tgt_feat_paths_sorted[0], delimiter=","))

            # now there is only one label file for each target proportion
            y_tgt = np.clip(np.loadtxt(tgt_label_paths_sorted[0]), 0, 1)

            X_tgt_comp = compound_descriptor(X_tgt, task_id, len(domains))

            # y_preds = est.predict(X_tgt_comp)
            y_preds = search.predict(X_tgt_comp)
            mae = mean_absolute_error(y_tgt, y_preds)
            print "Domain %s\tMAE = %2.4f" % (domains[task_id], mae)


def main_mtl_test_pooling():
    seed = sys.argv[1]
    dir = sys.argv[2]
    input_dir = "/home/desouza/Projects/qe_da/domains_large_bb17/" + dir + \
                "/stl_" + seed + "/"

    patt = "dom"
    src_feat_paths = glob.glob(input_dir + "features/*" + patt + "*.csv")
    src_label_paths = glob.glob(input_dir + "labels/*" + patt + "*.hter")

    if len(src_feat_paths) != len(src_label_paths):
        print "number of source feature files and label files differs: %d and " \
              "" \
              "" \
              "" \
              "%d" % (
                  len(src_feat_paths), len(src_label_paths))
        sys.exit(1)

    src_feat_paths_sorted = sorted(src_feat_paths)
    src_label_paths_sorted = sorted(src_label_paths)

    domains = ["it", "ted", "wsd"]
    # for proportion in range(len(src_feat_paths_sorted)):
    for proportion in [9]:
        print "# Proportion %s" % proportion
        X_src = np.loadtxt(src_feat_paths_sorted[proportion], delimiter=",")
        y_src = np.loadtxt(src_label_paths_sorted[proportion])

        # scales according to proportion
        src_scaler = StandardScaler().fit(X_src)
        X_src = src_scaler.transform(X_src)

        X_src_comp = compound_descriptor_at_once(X_src, y_src)

        est = PAMTLRegressor(17, 3)
        est.partial_fit(X_src, y_src)

        for task_id, domain in enumerate(domains):
            print "## Domain %s" % domain

            tgt_feat_paths = glob.glob(
                input_dir + "features/" + domain + "*.tgt.csv")
            tgt_label_paths = glob.glob(
                input_dir + "labels/" + domain + "*.tgt.hter")

            tgt_feat_paths_sorted = sorted(tgt_feat_paths)
            tgt_label_paths_sorted = sorted(tgt_label_paths)

            X_tgt = np.loadtxt(tgt_feat_paths_sorted[0], delimiter=",")
            y_tgt = np.loadtxt(tgt_label_paths_sorted[0])
            print "### Test on %s" % os.path.basename(tgt_feat_paths_sorted[0])

            X_tgt_comp = compound_descriptor(X_tgt, task_id, len(domains))
            y_preds = est.predict(X_tgt_comp)
            mae = mean_absolute_error(y_tgt, y_preds)
            print "MAE = %2.4f" % mae


if __name__ == "__main__":
    main_mtl_test()
    # main_mtl()
