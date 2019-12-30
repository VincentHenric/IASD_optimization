#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:19:36 2019

@author: henric
"""

import numpy as np

class Losses:
    @staticmethod
    def get(kind):
        pass
    
    @staticmethod
    def least_squares(X, y, w):
        return 1/(2*len(X)) * np.linalg.norm(X @ w - y)**2
        
    @staticmethod
    def logistic(X, y, w):
        return 1/len(X)*np.log(1+np.exp(-X@y))
    
class LeastSquare:
    @staticmethod
    def loss(X, y, w):
        return 1/(2*len(X)) * np.linalg.norm(X@w-y,2)**2
    
    @staticmethod
    def grad(X, y, w):
        return 1/len(X) * X.T @ (X@w-y)
    
    @staticmethod
    def L(X):
        return 1/len(X) * np.linalg.norm(X,2)
    
class Logistic:
    @staticmethod
    def loss(X, y, w):
        pass
    
    @staticmethod
    def grad(X, y, w):
        pass
    
    @staticmethod
    def L(X):
        return np.max(np.linalg.norm(X, axis=0))**2/(4*len(X))