# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:15:12 2015

@author: akusok
"""

import numpy as np
from all_methods import *


def viewc(E):
    cls_names = ["baseline",
                 "classification",
                 "classification_v",
                 "classification_cv",
                 "classification_loo",
                 "classification_mc_loo",
                 "classification_mc_cv",
                 "classification_wc_loo",
                 "classification_wc_cv"]
    E = E * 100
    E = E.reshape((9, -1))
#    E2 = E.mean(axis=2)
#    ix = E2.argmax(axis=1)
#    E = E[xrange(9), ix, :]
    m = E.mean(axis=1)
    s = E.std(axis=1)
    for i in xrange(9):
        print "%.02f : %.02f (%s)" % (m[i], s[i], cls_names[i])


def viewr(E):
    reg_names = ["baseline",
                 "regression",
                 "regression_v",
                 "regression_cv",
                 "regression_loo"]
    E = E.reshape((5, -1))
#    E2 = E.mean(axis=2)
#    ix = E2.argmax(axis=1)
#    E = E[xrange(5), ix, :]
    m = E.mean(axis=1)
    s = E.std(axis=1)
    for i in xrange(5):
        print "%.02f : %.02f (%s)" % (m[i], s[i], reg_names[i])


def Classification_Bench(fld, nn, ntp, rep, bs):
        E = np.zeros((9, rep, 10))
        E[0, :, :] = bs/100
        for i in xrange(rep):
            E[1, i, :] = classification(fld, nn, ntp)
            E[2, i, :] = classification_v(fld, nn, ntp)
            E[3, i, :] = classification_cv(fld, nn, ntp)
            E[4, i, :] = classification_loo(fld, nn, ntp)
            E[5, i, :] = classification_mc_loo(fld, nn, ntp)
            E[6, i, :] = classification_mc_cv(fld, nn, ntp)
            E[7, i, :] = classification_wc_loo(fld, nn, ntp)
            E[8, i, :] = classification_wc_cv(fld, nn, ntp)
        return E


def Regression_Bench(fld, nn, ntp, rep, bs):
        E = np.zeros((5, rep, 10))
        E[0, :, :] = bs
        for i in xrange(rep):
            E[1, i, :] = regression(fld, nn)
            E[2, i, :] = regression_v(fld, nn)
            E[3, i, :] = regression_cv(fld, nn)
            E[4, i, :] = regression_loo(fld, nn)
        return E


if __name__ == "__main__":
#    E = Classification_Bench("Classification-Iris", 15, "sigm", 3, 72.2)
#    E = Classification_Bench("Classification-Iris", 15, "lin", 10, 72.2)
#    E = Classification_Bench("Classification-Iris", 15, "tanh", 3, 72.2)
#    E = Classification_Bench("Classification-Iris", 15, "rbf_l2", 3, 72.2)
#    E = Classification_Bench("Classification-Iris", 15, "rbf_l1", 3, 72.2)
#    E = Classification_Bench("Classification-Iris", 15, "rbf_linf", 3, 72.2)
    E = Classification_Bench("Classification-Pima_Indians_Diabetes", 15, "sigm", 10, 72.2)
#    E = Classification_Bench("Classification-Wine", 15, "sigm", 10, 81.8)
#    E = Classification_Bench("Classification-Wisconsin_Breast_Cancer", 30, "sigm", 10, 95.6)    
    viewc(E)

#    E = Regression_Bench("Regression-Abalone",       20, "sigm", 10, 8.3) * 10 
#    E = Regression_Bench("Regression-Ailerons",     100, "sigm", 10, 3.3E-8) * 10**9
#    E = Regression_Bench("Regression-Auto_price",    20, "sigm", 10, 7.9E+9) * 10**-6 
#    E = Regression_Bench("Regression-Bank",         300, "sigm", 10, 6.7E-3) * 10**4
#    E = Regression_Bench("Regression-Boston",        70, "sigm", 10, 1.2E+2) * 10**0
#    E = Regression_Bench("Regression-Breast_cancer", 10, "sigm", 10, 7.7E+3) * 10**-2
#    E = Regression_Bench("Regression-Computer",     500, "sigm", 10, 4.9E+2) * 10**0
#    E = Regression_Bench("Regression-CPU",           40, "sigm", 10, 4.7E+4) * 10**-2
#    E = Regression_Bench("Regression-Elevators",     70, "sigm", 10, 2.2E-6) * 10**7
#    E = Regression_Bench("Regression-Servo",         40, "sigm", 10, 7.1) * 10**2
#    E = Regression_Bench("Regression-Stocks",       180, "sigm", 10, 3.4E+1) * 10**2
#    viewr(E)

