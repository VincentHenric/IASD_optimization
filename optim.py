#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:18:35 2019

@author: henric
"""
import numpy as np
import regularization

class IAST():
    """
    runs the ISTA algorithm
    """
    def __init__(self, loss, regul, lbda, **kwargs):
        self.loss = loss
        self.prox = regularization.Proximal_func.get(regul)
        self.regul = regularization.Regularization.get(regul)
        self.lbda = lbda
        self.regul_params = kwargs
        self.data = {}
        
    def single_step(self, X, y, w, tau):
        """
        performs a single iteration of the ISTA algorithm
        """
        grad_step = w - tau * self.loss.grad(X, y, w)
        return self.prox(grad_step, tau * self.lbda, **self.regul_params)
        
    def multiple_steps(self, X, y, w_init=None, tau=None, n=1, save=True):
        """
        performs n iterations of the ISTA algorithm
        """
        if not w_init:
            w_init = np.zeros((X.shape[1],1))
        if not tau:
            tau = 1/self.loss.L(X)
        if save:
            self.data['W'] = np.zeros((n, X.shape[1]))
            self.data['J'] = np.zeros(n)
            self.data['E'] = np.zeros(n)
        w = w_init
        for i in range(n):
            w = self.single_step(X, y, w, tau)
            if save:
                self.data['W'][i,:] = w.flatten()
                self.data['J'][i] = self.loss.loss(X,y,w)+self.regul(w, self.lbda, **self.regul_params)
                self.data['E'][i] = self.loss.loss(X,y,w)
        return w
