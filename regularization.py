#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:20:12 2019

@author: henric
"""
import numpy as np
from functools import partial

class Regularization:
#    @staticmethod
#    def get(kind, **kwargs):
#        if kind=='lasso':
#            return partial(Regularization.lasso(**kwargs))
#        elif kind=='ridge':
#            return partial(Regularization.ridge(**kwargs))
#        elif kind=='norm2':
#            return partial(Regularization.norm2(**kwargs))
#        elif kind=='elasticnet':
#            return partial(Regularization.elasticnet(**kwargs))
#        elif kind=='group_norm2':
#            return partial(Regularization.group_norm2(**kwargs))
    
    @staticmethod
    def get(kind, **kwargs):
        if kind=='lasso':
            return Regularization.lasso
        elif kind=='ridge':
            return Regularization.ridge
        elif kind=='norm2':
            return Regularization.norm2
        elif kind=='elasticnet':
            return Regularization.elasticnet
        elif kind=='group_norm2':
            return Regularization.group_norm2
    
    @staticmethod
    def lasso(x, lbda):
        return lbda * np.linalg.norm(x, ord=1)
    
    @staticmethod
    def ridge(x, lbda):
        return lbda / 2 * np.linalg.norm(x, ord=2)**2
    
    @staticmethod
    def norm2(x, lbda):
        return lbda * np.linalg.norm(x, ord=2)
    
    @staticmethod
    def elasticnet(x, lbda, gamma):
        return Regularization.lasso(x, lbda) + Regularization.ridge(x, lbda * gamma)
    
    @staticmethod
    def groupnorm2(x, lbda, groups):
        y = 0
        for group in groups:
            y += Regularization.norm2(x[group], lbda)
        return y
    
class Proximal_func:
#    @staticmethod
#    def get(kind, **kwargs):
#        if kind=='lasso':
#            return partial(Proximal_func.lasso(**kwargs))
#        elif kind=='ridge':
#            return partial(Proximal_func.ridge(**kwargs))
#        elif kind=='norm2':
#            return partial(Proximal_func.norm2(**kwargs))
#        elif kind=='elasticnet':
#            return partial(Proximal_func.elasticnet(**kwargs))
#        elif kind=='group_norm2':
#            return partial(Proximal_func.group_norm2(**kwargs))
    
    @staticmethod
    def get(kind, **kwargs):
        if kind=='lasso':
            return Proximal_func.lasso
        elif kind=='ridge':
            return Proximal_func.ridge
        elif kind=='norm2':
            return Proximal_func.norm2
        elif kind=='elasticnet':
            return Proximal_func.elasticnet
        elif kind=='group_norm2':
            return Proximal_func.group_norm2
    
    @staticmethod
    def lasso(x, lbda):
        return np.maximum(np.abs(x)-lbda, np.zeros(x.shape))*np.sign(x)
    
    @staticmethod
    def ridge(x, lbda):
        return 1/(1+lbda) * x
    
    @staticmethod
    def norm2(x, lbda):
        return np.maximum(1-lbda/np.linalg.norm(x), np.zeros(x.shape))*x
    
    @staticmethod
    def elasticnet(x, lbda, gamma):
        # lbda * ||x||1 + lbda * gamma / 2 * ||x||2²
        return 1/(1+lbda*gamma)*Proximal_func.lasso(x,lbda)
    
    @staticmethod
    def group_norm2(x, lbda, groups):
        y = np.zeros(x.shape)
        for group in groups:
            y[group]=Proximal_func.norm2(x[group],lbda)