# !/usr/bin/env python3
# EE4070 Numerical Analysis
# HW06 Conjugate Gradient Method
# 110071021, 吳睿芸
# Date: 2024/04/15

import numpy as np
import math

# in-place LU decomposition on n x n matrix A
def luFact(n, A):
    
    for i in range(n):
        # row copy 
        # a[i][j] to u[i][j] needs no action due to in-place LU
        
        # column division
        A[(i + 1) :, i] /= A[i][i]  # form l[j][i]
            
        # update lower submatrices
        for j in range(i + 1, n): 
            A[j, (i + 1) :] -= A[j][i] * A[i, (i + 1) :] 

    return A

# forward substitution, A = LU, return y, Ly = b
def fwdSubs(n, A, b):
    
    y = np.copy(b) # initialize y to b
    
    for i in range(n):
        y[i + 1: ] -= A[ (i + 1): ,i] * y[i]
    
    return y 

    
# backward substitution, A = LU, return x, Ux = y
def bckSubs(n, A, y) :
    
    x = np.copy(y) # initialize x to y
        
    for i in range(n - 1, -1, -1):
        x[i] /= A[i][i]
        
        for j in range(i - 1, -1, -1):
            x[j] -= A[j][i] * x[i]
    
    return x

# in-place Cholesky decomposition
def cholesky(n, A):
    
    for i in range(n):
        A[i][i] = math.sqrt(A[i][i]) # l[i][0] first column
        
        for j in range(i + 1, n):
            A[j][i] /= A[i][i]
        
        for j in range(i + 1, n):
            A[j, i + 1: j + 1] -= A[j][i] * A[i + 1: j + 1, i]
    
    return A

#  Cholesky forward and backward substitutions
def choSolve(n, A, b):
    
    # forward substitution, A = L@LT, return y, Ly = b
    y = np.copy(b) # initialize y to b
    for i in range(n):
        y[i] /= A[i][i]
        for j in range(i + 1, n):
            y[j] -= A[j][i] * y[i]
    
    # backward substitution, A = L@LT, return x, LT@x = y
    x = np.copy(y) # initialize x to y
    for i in range(n - 1, -1, -1):
        x[i] /= A[i][i]
        
        for j in range(i - 1, -1, -1):
            x[j] -= A[i][j] * x[i] # A[j][i] = A[i][j] for cholesky
    return x


def norm1(x): # Given an n-vector x return ∥x∥1.
    return sum(abs(x))
    
def norm2(x): # Given an n-vector x return ∥x∥2.
    return np.sqrt(sum(x**2))
    
def normInf(x): # Given an n-vector x return ∥x∥∞
    return max(abs(x))

            
def Jacobi(n, A, b, maxIter=1000000, tol=1e-7, enorm=norm1):
    D = np.zeros((n, n), dtype=float)
    E = np.copy(A)
    for i in range(n):
        D[i] = A[i,i]
    for i in range(n):
        E[i,i] = 0
    for i in (maxIter):
        x1 = (b[i] - E@x)/ D[i]
        if enorm(x-x1):
            return (True, x1)
        x = np.copy(x1)
    return (False, x)
    
def GS(n, A, b, maxIter=1000000, tol=1e-7, enorm=norm2):
    D = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    x1 = np.zeros(n, dtype=float)
    iter = 0
    for i in range(n):
        D[i] = A[i,i]
    
    for j in range(maxIter):
        for i in range(n):
            x1[i] = (b[i] - (A[i,:i] @ x1[:i]) - (A[i,i+1:] @ x[i+1:]) )/ D[i]
        if enorm(x-x1) < tol :
            iter += i
            return (True, x1, iter)
        x = np.copy(x1)
        iter = +i
    return (False, x, iter )
    
def SGS(n, A, b, maxIter=1000000, tol=1e-7, enorm=norm1):
    D = np.zeros(n, dtype=float)
    x = np.zeros(n, dtype=float)
    x1 = np.zeros(n, dtype=float)
    x2 = np.zeros(n, dtype=float)

    for i in range(n):
        D[i] = A[i,i]
    for i in range(n):
        x1[i] = (b[i] - A[i,:i] @ x1[:i] - A[i,i+1:] @ x[i+1:] )/ D[i]
    for i in range(maxIter):
        x2[i] = (b[i] - A[i,:i] @ x1[:i] - A[i,i+1:] @ x2[i+1:] )/ D[i]
        if enorm(x-x2) < tol :
            return (True, x1)
        x = np.copy(x2)
    return (False, x)

def CG(n, A, b, maxIter=1000000, tol=1e-8, enorm=normInf):
    x = np.zeros(n, dtype=float)
    x1 = np.zeros(n, dtype=float)
    p = np.copy(b)    # p0 = b - A@x = b
    r = np.copy(b)
    R = r.transpose() @ r

    for i in range(maxIter):
        Ap = A @ p     # Ap is A cross p
                 
        ak = R / ( p.transpose() @ Ap )  # d = pT @ A @ p
        x1 = x + ak*p
        r = r - ak*Ap  # r(k+1)= r(k)- αkAp(k)

        R1 = r.transpose() @  r   # R1 = r(k+1) @ r(k+1)
        
        bk = R1 / R    # bk = r(k+1) @ r(k+1) / r(k) @ r(k)
        p = r + bk*p   # p(k+1)= r(k+1)− βk*p(k)

        if enorm(x-x1) < tol :
            return (True, i+1, x1) # i+1: number of iterations performed
        
        x = np.copy(x1)
        R = np.copy(R1)
    return (False, maxIter, x)
