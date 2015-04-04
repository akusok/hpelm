# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:48:33 2014

@author: akusok
"""

from elm_base import ELM_BASE
from mss_loo import MSS_LOO
from mss_v import MSS_V
from mss_cv import MSS_CV


class ELM(ELM_BASE):
    """Interface for training Extreme Learning Machines.
    """

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
        :param "kmax": maximum number of neurons (with "OP")

        System setup
        :param "classification"/"c": build ELM for classification
        :param "classification balanced"/"cb": build ELM with balanced classification
        :param "multiclass"/"mc": build ELM for multiclass classification
        :param "adaptive"/"ad": build adaptive ELM for non-stationary model
        :param "batch": batch size for adaptive ELM (sliding window step size)
        """

        assert len(self.neurons) > 0, "Add neurons to ELM before training it"
        X, T = self._checkdata(X, T)
        args = [a.upper() for a in args]  # make all arguments upper case

        # kind of "enumerators", try to use only inside that script
        MODELSELECTION = None  # V / CV / MCCV / LOO / None
        ADAPTIVE = False  # batch / None
        batch = None

        # check exclusive parameters
        assert len(set(args).intersection(set(["V", "CV", "LOO"]))) <= 1, "Use only one of V / CV / LOO"
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
                assert k >= 3, "Use at least k=3 splits for Cross-Validation"
                MODELSELECTION = "CV"
            if a == "LOO":
                MODELSELECTION = "LOO"
            if a == "OP":
                self.ranking = "OP"
                if "kmax" in kwargs.keys():
                    self.kmax = int(kwargs["kmax"])
            if a in ("C", "CL", "CLASSIFICATION"):
                self.classification = "c"
            if a in ("CB", "BC", "CLASSIFICATION BALANCED", "BALANCED CLASSIFICATION"):
                self.classification = "cb"
            if a in ("MC", "MULTICLASS"):
                self.classification = "mc"
            # if a in ("A", "AD", "ADAPTIVE"):
            #     assert "batch" in kwargs.keys(), "Provide batch size for adaptive ELM model (batch)"
            #     batch = kwargs['batch']
            #     ADAPTIVE = True

        # polymorphism switching to a correct "_train" method
        if MODELSELECTION == "V":
            self.__class__ = MSS_V
            self._train(X, T, Xv, Tv)
        elif MODELSELECTION == "CV":
            self.__class__ = MSS_CV
            self._train(X, T, k)
        elif MODELSELECTION == "LOO":
            self.__class__ = MSS_LOO
            self._train(X, T)
        else:
            self._train(X, T)
        self.__class__ = ELM





























