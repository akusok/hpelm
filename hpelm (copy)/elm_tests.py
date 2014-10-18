'''
Created on Aug 18, 2014

@author: akusoka1
'''
import numpy as np
from numpy.testing import assert_allclose
import unittest

#from elm import ELM
from d_elm import DELM as ELM
from elm_error import ELMError


class Test_Neurons(unittest.TestCase):
    """Test set for ELM neurons configuration.
    """
    
    def test_Init_SetInputsTargets_ModelHasFixedInputsAndTargets(self):
        elm = ELM(5, 2)
        self.assertEqual(elm.inputs, 5)
        self.assertEqual(elm.targets, 2)
    
    def test_Init_SetInputsTargetsKeywords_ModelHasFixedInputsAndTargets(self):
        elm = ELM(inputs=5, targets=2)
        self.assertEqual(elm.inputs, 5)
        self.assertEqual(elm.targets, 2)

    def test_AddNeurons_AddNeuronsSimple_HasNeurons(self):
        elm = ELM(1,1)
        elm.add_neurons(3, np.tanh)
        self.assertEqual(len(elm.ufunc), 3)
        self.assertIs(elm.ufunc[0], np.tanh)    
    
    def test_AddNeurons_AddNeuronsSeveralTimes_HasAllNeurons(self):
        elm = ELM(1,1)
        elm.add_neurons(3, np.tanh)
        elm.add_neurons(2, np.tanh)
        self.assertEqual(len(elm.ufunc), 5)
        correct_neuron_types = [e0 is np.tanh for e0 in elm.ufunc]
        self.assertTrue(np.all(correct_neuron_types)) 

    def test_AddNeurons_AddNeuronsTypeNone_HasLinearNeurons(self):
        elm = ELM(1,1)
        elm.add_neurons(3, None)
        identity_func = elm.ufunc[0]
        a = 1.35
        self.assertEqual(a, identity_func(a))    
    
    def test_AddNeurons_AddNeuronsMultipleTypes_HasAllNeurons(self):
        elm = ELM(1,1)
        elm.add_neurons(3, np.tanh)
        elm.add_neurons(2, np.sinh)
        self.assertEqual(len(elm.ufunc), 5)
        tanh_neurons = [e0 is np.tanh for e0 in elm.ufunc]
        sinh_neurons = [e0 is np.sinh for e0 in elm.ufunc]
        self.assertEqual(sum(tanh_neurons), 3)
        self.assertEqual(sum(sinh_neurons), 2)
    
    def test_AddNeurons_AddNeuronsParam_HasParameters(self):
        elm = ELM(5,2)
        W = np.random.randn(5,3)
        bias = np.random.rand(3)
        elm.add_neurons(3, np.tanh, W, bias)
        assert_allclose(elm.W, W)
        assert_allclose(elm.bias, bias)
    
    def test_AddNeurons_AddNeuronsParamMultiple_ParametersCorrectlyCombined(self):
        elm = ELM(3,2)
        W = np.random.randn(3,5)
        bias = np.random.rand(5)
        elm.add_neurons(3, np.tanh, W[:,:3], bias[:3])
        elm.add_neurons(3, np.tanh, W[:,3:], bias[3:])
        assert_allclose(elm.W, W)
        assert_allclose(elm.bias, bias)
    
    def test_AddNeurons_AddSingleNeuron_CorrectParameterMerge(self):
        elm = ELM(3,2)
        elm.add_neurons(1, np.tanh)
        elm.add_neurons(1, np.tanh)
        W = np.random.randn(3,1)
        bias = np.random.rand(1)
        elm.add_neurons(1, np.tanh, W, bias)
        elm.add_neurons(1, np.tanh, W, bias)
        self.assertEqual(len(elm.ufunc), 4)
        self.assertSequenceEqual(elm.W.shape, (3,4))
    
    def test_AddNeurons_CustomTransformationFunction_CorrectlySaved(self):
        elm = ELM(3,2)
        my_func = np.sinh
        elm.add_neurons(3, my_func)
        self.assertIs(elm.ufunc[0], my_func)
        


class Test_BasicELM(unittest.TestCase):
    """Test set for ELM neurons configuration.
    """        
        
    def test_Train_ProjectDataNoBias_CorrectProjection(self):
        elm = ELM(5,2)
        X = np.random.randn(20,5)
        W = np.random.randn(5,7)
        bias = np.zeros((7,))
        elm.add_neurons(7, None, W, bias)
        H = elm._project(X)
        assert_allclose(H, X.dot(W))

    def test_Train_ProjectDataWithBias_CorrectProjection(self):
        elm = ELM(5,2)
        X = np.random.randn(20,5)
        W = np.random.randn(5,7)
        bias = np.random.rand(7)
        elm.add_neurons(7, None, W, bias)
        H = elm._project(X)
        assert_allclose(H, X.dot(W) + bias.reshape(1,7))

    def test_Run_ProblemXOR_SolvedWithSingleNeuron(self):
        X = np.array([[0,0], [1,1], [1,0], [0,1]])
        Y = np.array([1,1,-1,-1])
        for _ in range(100):
            try:
                elm = ELM(2,1)
                elm.add_neurons(1, np.tanh)
                elm.train(X,Y)
                Yh = elm.run(X)
                self.assertGreater(Yh[0], 0)
                self.assertGreater(Yh[1], 0)
                self.assertLess(Y[2], 0)
                self.assertLess(Y[3], 0)
                return
            except:
                pass
        self.fail("Cannot train 1 neuron to solve XOR problem in 100 re-initializations")

    def test_Train_TargetsOneTwoDimensions_WorksFine(self):
        X = np.random.randn(10,3)
        T1 = np.random.rand(10)
        T2 = T1.reshape((10,1))
        elm = ELM(3,1)
        elm.add_neurons(5, np.tanh)
        elm.train(X, T1)
        Th1 = elm.run(X)
        elm.train(X, T2)
        Th2 = elm.run(X).ravel()
        assert_allclose(Th1, Th2)
        
    def test_Train_InputsAndTargetsDifferentNumberOfSamples_RaisesError(self):
        X = np.random.randn(10,3)
        T = np.random.randn(11)
        elm = ELM(3,1)
        elm.add_neurons(3, np.tanh)
        with self.assertRaises(ELMError):
            elm.train(X,T)
        


class Test_MPI_ELM(unittest.TestCase):
    """Test MPI distributed ELM functionality.
    """    
    
    def test_Project_DistributedProjection_CorrectH(self):
        pass



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test']
    unittest.main()







































