import glob
import os
import sys

import numpy as np
from sklearn.grid_search import RandomizedSearchCV
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.metrics import mean_absolute_error
import scipy.io as sio
from sklearn.utils import shuffle

from mtl.omtl import compound_descriptor_at_once, compound_descriptor
from mtl.partl_regression import PARTLRegressor
from stl.pa_regression import PARegressor
import scipy as sp


__author__ = 'desouza'


def main():
    n_jobs = 8
    n_iter = 100
    n_iter_rtl = 100
    cv_folds = 5
    shuffles = int(sys.argv[1])
    start = int(sys.argv[2])
    work_dir = sys.argv[3]
    domains = ["e-learning", "mf", "it", "ted"]

    for seed in range(start, start + shuffles, 1):
        # print "### ", seed
        input_dir = work_dir + os.sep + "matlab_" + str(seed) + os.sep

        glob_path = input_dir + os.sep + "mats/*.mat"
        # print glob_path
        data_sets = sorted(glob.glob(glob_path))
        # takes only 100% of training
        # data_sets = [data_sets[9]]

        eval_dir = input_dir + os.sep + "evaluation"
        pred_dir = input_dir + os.sep + "predictions"

        models_perform = {}
        models_params = {}
        # models_names = ["pamtl-ipl", "pamtl-half", "partl-vn",
        #                 "partl-ld", "pooling", "pa"]
        models_names = ["pamtl-ipl", "partl-vn",
                        "partl-ld", "pooling", "pa"]

        for model_name in models_names:
            dom_perf = {}
            dom_params = {}
            for dom in domains:
                dom_perf[dom] = np.zeros(len(data_sets))
                dom_params[dom] = []

            dom_perf["all"] = np.zeros(len(data_sets))
            dom_params["all"] = []

            models_perform[model_name] = dom_perf
            models_params[model_name] = dom_params

        # 10 different training set sizes
        for prop, mat_path in enumerate(data_sets):
            mat = sio.loadmat(mat_path)
            src_feats = mat['x_train_cell']
            src_labs = mat['y_train_cell']
            tgt_feats = mat['x_test_cell']
            tgt_labs = mat['y_test_cell']


            # for online mtl models
            # adds bias
            src_bias = []
            tgt_bias = []
            for i in xrange(len(src_feats[0])):
                X_train = src_feats[0][i]
                X_train = np.column_stack(
                    (X_train, np.ones((X_train.shape[0], 1))))
                src_bias.append(X_train)
                X_test = tgt_feats[0][i]
                X_test = np.column_stack(
                    (X_test, np.ones((X_test.shape[0], 1))))
                tgt_bias.append(X_test)

            Xc_train, yc_train = compound_descriptor_at_once(
                src_bias, src_labs[0].tolist())
            yc_train = yc_train.ravel()
            Xc_test, yc_test = compound_descriptor_at_once(
                tgt_bias, tgt_labs[0].tolist())
            yc_test = yc_test.ravel()

            Xc_train, yc_train = shuffle(Xc_train, yc_train, random_state=seed)

            comp_feats_test = []
            comp_labs_test = []
            for dom_id in xrange(len(domains)):
                comp_feats_test.append(
                    compound_descriptor(tgt_bias[dom_id], dom_id, len(domains)))
                comp_labs_test.append(tgt_labs[0][dom_id])

            print "train", Xc_train.shape
            print "test", Xc_test.shape
            print

            ### for pooling
            X_train = np.row_stack(tuple(src_bias))
            y_train = yc_train
            X_test = np.row_stack(tuple(tgt_bias))
            y_test = yc_test

            X_train, y_train = shuffle(X_train, y_train, random_state=seed)

            pa_pooling = PARegressor(fit_intercept=False, loss="pai", n_iter=1)
            np.random.seed(seed)
            param_dist = {"C": sp.stats.expon(scale=10),
                          "epsilon": sp.stats.expon(scale=.1),
                        }
            search_pooling = RandomizedSearchCV(pa_pooling, param_dist,
                                                n_iter=n_iter,
                                                scoring="mean_absolute_error",
                                                n_jobs=n_jobs, iid=False,
                                                refit=True,
                                                cv=cv_folds, verbose=1,
                                                random_state=seed)
            search_pooling.fit(X_train, y_train)
            print search_pooling.best_params_
            y_preds = np.clip(search_pooling.predict(X_test), 0, 1)
            predfname = pred_dir + os.sep + "pa-pooling." + str(
                prop) + ".all.preds"
            np.savetxt(predfname, y_preds, fmt="%2.8f")
            modfname = pred_dir + os.sep + "pa-pooling." + str(
                prop) + ".all.model"
            np.savetxt(modfname, search_pooling.best_estimator_.coef_)

            mae = mean_absolute_error(y_test, y_preds)
            print "PA pooling mae =", mae

            models_perform["pooling"]["all"][prop] = mae
            models_params["pooling"]["all"].append(search_pooling.best_params_)


            training_size = Xc_train.shape[0]
            rounds = int(0.5 * training_size)
            ### PARTL-vn
            partl_vn = PARTLRegressor(len(domains), 18, divergence="vn",
                                      centered=True, loss="pai", n_iter=1,
                                      rounds_num=rounds)

            np.random.seed(seed)
            param_dist = {"C": sp.stats.expon(scale=10),
                  "epsilon": sp.stats.expon(scale=.1),
                  "eta": sp.stats.expon(scale=10)
                }
            search_vn = RandomizedSearchCV(partl_vn, param_dist,
                                           n_iter=n_iter_rtl,
                                           scoring="mean_absolute_error",
                                           n_jobs=n_jobs, iid=False, refit=True,
                                           cv=cv_folds, verbose=1,
                                           random_state=seed)
            search_vn.fit(Xc_train, yc_train)
            print search_vn.best_params_
            y_preds = np.clip(search_vn.predict(Xc_test), 0, 1)
            predfname = pred_dir + os.sep + "partl-vn." + str(
                prop) + ".all.preds"
            np.savetxt(predfname, y_preds, fmt="%2.8f")
            modfname = pred_dir + os.sep + "partl-vn." + str(
                prop) + ".all.model"
            np.savetxt(modfname, search_vn.best_estimator_.coef_)

            mae = mean_absolute_error(yc_test, y_preds)
            print "PARTL-vn mae =", mae

            models_perform["partl-vn"]["all"][prop] = mae
            models_params["partl-vn"]["all"].append(search_vn.best_params_)


            # ### PARTL-ld
            partl_ld = PARTLRegressor(len(domains), 18, divergence="ld",
                                      centered=True, loss="pai", n_iter=1,
                                      rounds_num=rounds)
            np.random.seed(seed)
            param_dist = {"C": sp.stats.expon(scale=10),
                  "epsilon": sp.stats.expon(scale=.1),
                  "eta": sp.stats.expon(scale=10)
                }
            search_ld = RandomizedSearchCV(partl_ld, param_dist,
                                           n_iter=n_iter_rtl,
                                           scoring="mean_absolute_error",
                                           n_jobs=n_jobs, iid=False, refit=True,
                                           cv=cv_folds, verbose=1,
                                           random_state=seed)
            search_ld.fit(Xc_train, yc_train)
            print search_ld.best_estimator_
            print search_ld.best_params_
            y_preds = np.clip(search_ld.predict(Xc_test), 0, 1)
            predfname = pred_dir + os.sep + "partl-ld." + str(
                prop) + ".all.preds"
            np.savetxt(predfname, y_preds, fmt="%2.8f")
            modfname = pred_dir + os.sep + "partl-ld." + str(
                prop) + ".all.model"
            np.savetxt(modfname, search_ld.best_estimator_.coef_)

            mae = mean_absolute_error(yc_test, y_preds)
            print "PARTL-ld mae =", mae

            models_perform["partl-ld"]["all"][prop] = mae
            models_params["partl-ld"]["all"].append(search_ld.best_params_)

            ### for online STL
            #
            # param_dist = {"C": [10 ** i for i in np.arange(-10, 10).tolist()],
            # "epsilon": [10 ** i for i in
            #                           np.arange(-20, 0).tolist()],
            #               "loss": ["epsilon_insensitive",
            # "squared_epsilon_insensitive"]}
            #
            np.random.seed(seed)
            param_dist = {"C": sp.stats.expon(scale=10),
                          "epsilon": sp.stats.expon(scale=.1),
                        }

            for dom_id, dom_name in enumerate(domains):
                # pa = PassiveAggressiveRegressor(n_iter=1, fit_intercept=False, loss="epsilon_insensitive")
                pa = PARegressor(fit_intercept=False, n_iter=1, loss="pai")
                search_pa = RandomizedSearchCV(pa, param_dist, n_iter=n_iter,
                                               scoring="mean_absolute_error",
                                               n_jobs=n_jobs, iid=True,
                                               refit=True, cv=cv_folds,
                                               verbose=1, random_state=seed)
                x_train = np.column_stack((src_feats[0][dom_id], np.ones(
                    (src_feats[0][dom_id].shape[0], 1))))
                search_pa.fit(x_train, src_labs[0][dom_id].ravel())
                print search_pa.best_params_
                x_test = np.column_stack((tgt_feats[0][dom_id], np.ones(
                    (tgt_feats[0][dom_id].shape[0], 1))))
                y_preds = np.clip(search_pa.predict(x_test), 0, 1)
                predfname = pred_dir + os.sep + "pa." + str(
                    prop) + "." + dom_name + ".preds"
                np.savetxt(predfname, y_preds, fmt="%2.8f")
                modfname = pred_dir + os.sep + "pa." + str(
                    prop) + "." + dom_name + ".model"
                np.savetxt(modfname, search_pa.best_estimator_.coef_)

                mae = mean_absolute_error(tgt_labs[0][dom_id].ravel(), y_preds)
                print "PA STL %s = %2.5f" % (dom_name, mae)
                models_perform["pa"][dom_name][prop] = mae
                models_params["pa"][dom_name].append(search_pa.best_params_)

                y_preds = np.clip(search_pooling.predict(x_test), 0, 1)
                predfname = pred_dir + os.sep + "pa-pooling." + str(
                    prop) + "." + dom_name + ".preds"
                np.savetxt(predfname, y_preds, fmt="%2.8f")

                mae = mean_absolute_error(tgt_labs[0][dom_id].ravel(), y_preds)
                print "Pooling %s = %2.5f" % (dom_name, mae)
                models_perform["pooling"][dom_name][prop] = mae

            print "### In-domain eval:"
            feats_test = comp_feats_test
            labs_test = comp_labs_test
            # models = [search_ipl, search_half, search_vn, search_ld]

            models = [search_ipl, search_vn, search_ld]

            for dom, X_test, y_test in zip(domains, feats_test, labs_test):
                print dom
                for est_name, est in zip(models_names, models):
                    y_preds = np.clip(est.predict(X_test), 0, 1)
                    predfname = pred_dir + os.sep + est_name + "." + str(
                        prop) + "." + dom + ".preds"
                    np.savetxt(predfname, y_preds, fmt="%2.8f")

                    mae = mean_absolute_error(y_test.ravel(), y_preds)
                    print "%s = %2.5f" % (est_name, mae)

                    models_perform[est_name][dom][prop] = mae

            print

        # print models_perform
        for model_name, domains_dict in models_perform.items():
            # print model_name
            for domain_name, performances in domains_dict.items():
                eval_name = eval_dir + os.sep + model_name.lower() + "." + \
                            domain_name + ".mae"
                np.savetxt(eval_name, performances)
                # print domain_name, performances

        # print models_params
        for model_name, domains_dict in models_params.items():
            for domain_name, params in domains_dict.items():
                params_name = eval_dir + os.sep + model_name.lower() + "." + \
                              domain_name + ".params"
                params_file = open(params_name, "w")
                params_file.write(str(params))
                params_file.close()


if __name__ == "__main__":
    main()
