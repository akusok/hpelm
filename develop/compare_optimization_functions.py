# -*- coding: utf-8 -*-
"""

COMPARED BUILD-IN SINGLE VARIABLE OPTIMIZATION FUNCTION AND MY OWN IMPLEMENTATION.

MY IMPLEMENTATION CREATES 50% MORE FUNCTION CALLS.

MY IMPLEMENTATION PERFORMS BETTER BECAUSE IT CHECKS MAXIMUM AND MINIMUM NUMBERS OF NEURONS ALSO, WHILE 
BUILD-IN IMPLEMENTATION CHECKS ROUGHLY 10% AND 85% BORDERS AND DOES NOT CHECK OUTSIDE THEM IF THERE IS
A LOCAL MINIMA INSIDE THAT REGION.


"""

import numpy as np
from numpy.linalg import pinv
from scipy.optimize import minimize_scalar, brenth

from slfn import SLFN


class ELM(SLFN):
    """Non-parallel Extreme Learning Machine.
    """

    # inherited  def _checkdata(self, X, T):
    # inherited  def add_neurons(self, number, func, W=None, B=None):
    # inherited  def project(self, X):
    # inherited  def predict(self, X):
    # inherited  def save(self, model):
    # inherited  def load(self, model):

    def __init__(self, inputs, outputs):
        """Universal contructor of ELM model.

        :param neurons: number of neurons or exact neuron
        """
        super(ELM, self).__init__(inputs, outputs)

    def train(self, X, T, *args, **kwargs):
        """Universal training interface for ELM model with model structure selection.

        :param X: input data matrix
        :param T: target data matrix

        Model structure selection (exclusive, choose one)
        :param "V": use validation set
        :param "CV": use cross-validation
        :param "LOO": use leave-one-out validation

        Additional parameters for model structure selecation
        :param Xv: validation data X ("V")
        :param Tv: validation targets T ("V")
        :param k: number of splits ("CV")

        Ranking of hidden neurons
        :param "HQ": use Hannan-Quinn criterion
        :param "OP": use Optimal Pruning (OP-ELM)

        System setup
        :param "classification"/"c": build ELM for classification
        :param "multiclass"/"mc": build ELM for multiclass classification
        :param "adaptive"/"ad": build adaptive ELM for non-stationary model
        :param "batch": batch size for adaptive ELM (sliding window step size)
        """

        assert len(self.neurons) > 0, "Add neurons to ELM before training it"
        X, T = self._checkdata(X, T)
        args = [a.upper() for a in args]  # make all arguments upper case

        # kind of "enumerators", try to use only inside that script
        MODELSELECTION = None  # V / CV / MCCV / LOO / None
        NEURONRANKING = None  # HQ / OP / None
        CLASSIFICATION = None  # c / mc / None
        ADAPTIVE = False  # batch / None
        Xv = None
        Tv = None
        k = None
        batch = None

        # check exclusive parameters
        assert len(set(args).intersection(set(["V", "CV", "MCCV", "LOO"]))) <= 1, "Use only one of V / CV / MCCV / LOO"
        assert len(set(args).intersection(set(["HQ", "OP"]))) <= 1, "Use only one of HQ / OP"
        assert len(set(args).intersection(set(["C", "MC"]))) <= 1, "Use only one of classification / multiclass (c / mc)"

        # parse parameters
        for a in args:
            if a == "V":  # validation set
                assert "Xv" in kwargs.keys(), "Provide validation dataset (Xv)"
                assert "Tv" in kwargs.keys(), "Provide validation targets (Tv)"
                Xv = kwargs['Xv']
                Tv = kwargs['Tv']
                Xv, Tv = self._checkdata(Xv, Tv)
                MODELSELECTION = "V"
            if a == "CV":
                assert "k" in kwargs.keys(), "Provide Cross-Validation number of splits (k)"
                k = kwargs['k']
                MODELSELECTION = "CV"
            if a == "LOO":
                MODELSELECTION = "LOO"
            if a == "HQ":
                NEURONRANKING = "HQ"
            if a == "OP":
                NEURONRANKING = "OP"
            if a in ("C", "CL", "CLASSIFICATION"):
                CLASSIFICATION = "c"
            if a in ("MC", "MULTICLASS"):
                CLASSIFICATION = "mc"
            if a in ("A", "AD", "ADAPTIVE"):
                assert "batch" in kwargs.keys(), "Provide batch size for adaptive ELM model (batch)"
                batch = kwargs['batch']
                ADAPTIVE = True

        if MODELSELECTION == "V":
            self._train_v(X, T, Xv, Tv)
        else:
            self.Beta = self._solve(self.project(X), T)

    def _project_corr(self, X, T):
        """Create correlation matrices of projected data and targets.
        """
        H = self.project(X)
        HH = np.dot(H.T, H)
        HT = np.dot(H.T, T)
        return HH, HT

    def _solve(self, H, T):
        """Solve a linear system.
        """
        P = pinv(H)
        Beta = np.dot(P, T)
        return Beta

    def _error(self, Y, T):
        """Return an error for given Y and T.

        Differs for classification and multiclass.
        """
        err = np.mean((Y - T)**2)
        return err

    def _prune(self, idx):
        """Leave only neurons with the given indexes.
        """
        idx = list(idx)
        neurons = []
        for nold in self.neurons:
            k = nold[1]  # number of neurons
            ix1 = [i for i in idx if i < k]  # index for current neuron type
            idx = [i-k for i in idx if i >= k]
            func = nold[0]
            number = len(ix1)
            W = nold[2][:, ix1]
            bias = nold[3][ix1]
            neurons.append((func, number, W, bias))
        self.neurons = neurons

    def _train_v(self, X, T, Xv, Tv):
        HH, HT = self._project_corr(X, T)
        Hv = self.project(Xv)
        nn = Hv.shape[1]
        errors = np.ones((nn,)) * -1

        rank = np.arange(nn)  # create ranking of neurons
        np.random.shuffle(rank)

        Beta = self._solve(HH, HT)
        Yv = np.dot(Hv, Beta)
        penalty = self._error(Yv, Tv)*0.01 / nn

        def error_v(k, errors, rank, HH, HT, Hv, Tv, penalty):
            if errors[k] == -1:
                rank1 = rank[:k]
                HH1 = HH[rank1, :][:, rank1]
                HT1 = HT[rank1, :]
                print k, HH1.shape, HT1.shape, k, len(rank)
                B = self._solve(HH1, HT1)
                Yv = np.dot(Hv[:, rank1], B)
                errors[k] = self._error(Yv, Tv) + k*penalty
            return errors[k]

        # this works really good! same result with less re-calculations
        result = minimize_scalar(error_v,
                                 bounds=(1, nn),
                                 args=(errors, rank, HH, HT, Hv, Tv, penalty),
                                 method="Bounded",
                                 tol=0.5)
        print result
        best_nn = rank[:result.x]
        from matplotlib import pyplot as plt
        plt.plot([i for i in range(nn) if errors[i]>0], [e for e in errors if e > 0], "*b")

        e2 = np.ones((nn,)) * -1
        e3 = np.ones((nn,)) * -1
        for k in xrange(1,nn):
            rank1 = rank[:k]
            HH1 = HH[rank1, :][:, rank1]
            HT1 = HT[rank1, :]
            B = self._solve(HH1, HT1)
            Yv = np.dot(Hv[:, rank1], B)
            errors[k] = self._error(Yv, Tv)
            e2[k] = self._error(Yv, Tv)
            e3[k] = e2[k] + k*penalty

        plt.plot(range(1,nn), e2[1:], '-k')
        plt.plot(range(1,nn), e3[1:], '-r')
        
        
        # MYOPT function
        # [A  B  C  D  E] interval points,
        # halve the interval each time

        # init part
        e = np.ones((nn,)) * -1  # error for all numbers of neurons
        er = np.ones((nn,)) * -1  # error for all numbers of neurons
        A = 1
        E = nn-1
        l = E - A
        B = A + l/4
        C = A + l/2
        D = A + 3*l/4

        while True:
            for idx in [A, B, C, D, E]:  # calculate errors at points
                if e[idx] == -1:
                    e[idx] = error_v(idx, er, rank, HH, HT, Hv, Tv, penalty)
            m = min(e[A], e[B], e[C], e[D], e[E])  # find minimum element
            if m in (e[A], e[B]):  # halve the search interval
                E = C
                C = B
            elif m in (e[D], e[E]):
                A = C
                C = D
            else:
                A = B
                E = D
            l = E - A
            B = A + l/4
            D = A + (3*l)/4
            # minimum is found
            if l < 3:
                break

        k_opt = [k for k in [A, B, C, D, E] if e[k] == m][0]  # find minimum index
        best_nn = rank[:k_opt]        
        
        plt.plot([i for i in range(nn) if e[i]>0], [e1 for e1 in e if e1 > 0], "dm")

        self._prune(best_nn)
        self.Beta = self._solve(self.project(X), T)
        print "%d of %d neurons selected with a validation set" % (len(best_nn), nn)
        if len(best_nn) > nn*0.9:
            print "Hint: try re-training with more hidden neurons"

        plt.show()




































