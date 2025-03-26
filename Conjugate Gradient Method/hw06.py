#!/usr/bin/env python3
# EE4070 Numerical Analysis
# HW06 Conjugate Gradient Method
# 110071021, 吳睿芸
# Date: 2024/04/15

import numpy as np
import time
from ee4070 import *

def Radd(A, rp, rn, g):
	''' Add resistor with conductance g to circuit matrix A
        rp: positive node; rn: negative node '''
	A[rp, rp] += g
	A[rn, rn] += g
	A[rp, rn] -= g
	A[rn, rp] -= g

def Vsrc(n, A, b, vp, V):
	''' Add grounded votage source to circuit matrix A and RHS b
        vp: positive node for the voltage source
        V: voltage value
		This creates a asymmetric matrix'''
	for i in range(n):
		A[vp, i] = 0
	A[vp, vp] = 1
	b[vp] = V

def VsrcSym(n, A, b, vp, V):
	''' Add grounded votage source to circuit matrix A and RHS b
        vp: positive node for the voltage source
        V: voltage value
		This creates a symmetric matrix'''
	Vsrc(n, A, b, vp, V)
	for i in range(n):
		if A[i, vp] != 0 and i != vp:
			b[i] -= A[i, vp] * V
			A[i, vp] = 0

def Mat(n):
    ''' create simple resistor network system matrix'''
    for i in range(n): # n nodes in system
        rn = i
        ''' every node connect to the node on the right side
            except for the nodes on the last column '''
        if (rn + 1) <= (n - n_perside):
            rp = rn + n_perside # find the node on the right side
            Radd(A, rp, rn, g)  # Add resistor
        
        ''' every node connect to the node below it
            except for the nodes on the last row '''
        if (i + 1) % n_perside != 0:
            rp = rn + 1        # find the node below it
            Radd(A, rp, rn, g) # Add resistor

    # set grounded and grounded votage source to 0 and V
    grounded(n_perside, n) 

def grounded(n_perside, n):
    ''' grounded 
        and grounded votage source '''
    # grounded at node (n_perside + n)/ 2
    # count from zero , so minus 1
    vp = int ( (n_perside + n)/ 2 - 1)
    for i in range(n):
        A[vp, i] = 0
    A[vp, vp] = 1

    # grounded votage source is at node 1
    vp = 0      # count from zero, 1 - 1
    VsrcSym(n, A, b, vp, V)

def get_Vw(N, x):
    # Vw is node number N/2 + 1
    w = (N/2 + 1) - 1       # count from zero, minus 1
    Vw = x[int(w)]          # Vw is x[(N/2 - 1) - 1]
    Vw = round(Vw, 6)       # Round to the sixth decimal place
    return Vw

def get_Vne(N, x): 
    # Vne is node number n - N
    ne = n - N - 1          # count from zero, minus 1
    Vne = x[int(ne)]        # Vne is x[n - N - 1]
    Vne = round(Vne, 6)     # Round to the sixth decimal place
    return Vne

def get_Vne(N, x):
    # Vne is node number n - N/2
    e = n - N/2 - 1         # count from zero, minus 1
    Ve = x[int(e)]          # Ve is x[n - N/2 - 1]
    Ve = round(Ve, 6)       # Round to the sixth decimal place
    return Ve

def r_equal(g, x, N, V):
    ''' calculate equivant resistance in this network '''
    ''' take node1(x[0])
        calculate total current for resistor network'''
    # current from node 1 to node 2
    i1 = (x[0] - x[1]) * g
    
    # current from node 1 to node 2
    i2 = (x[0] - x[N + 1]) * g
    
    I = i1 + i2     # total current for resistor network
    Req = V / I     # Req = V total / I total
    return Req

# read str for Number of resistors per side
str = input('Number of resistors per side :')
N = int(str)         # Number of resistors per side
print('N = ', N)     # print N
n = (N + 1) ** 2     # n : total amount of nodes
n_perside = N + 1    # n_perside: total amount of nodes perside
g = 1 / (2000 / N)   # resistant = 2000 / N, convert to g
V = 1                # the voltage V is 1 volt


# calculate CPU time of solving LU decomposition
A = np.zeros((n, n), dtype=float)  # create an nxn zero matrix A
b = np.zeros(n, dtype=float)   # create an nx1 zero vector b
Mat(n)
t0 = time.time()     # record time to t0
A = luFact(n, A)     # LU decomposition
y = fwdSubs(n, A, b) # forward substitution   Ly = b
x_LU = bckSubs(n, A, y) # backward substitution  Ux = y
t0 = (time.time() - t0)    # calculate CPU time
print('LU Decomposition CPU time = {:g} seconds'.format(t0)) # print out CPU time

# calculate CPU time of Conjugate Gradient Method
A = np.zeros((n, n), dtype=float) # create an nxn zero matrix A
b = np.zeros(n, dtype=float)      # create an nx1 zero vector b
Mat(n)
t0 = time.time()                  # record time to t0
x_CG = CG(n, A, b)                # Conjugate Gradient Method
t0 = (time.time() - t0)           # calculate CPU time
print("CG CPU time = {:g} seconds".format(t0)) # print out CPU time

print("Iter:", x_CG[1])
print('accuracy of CG vs. LU: ', norm2( x_LU - x_CG[2]) )
# print("error of CG: ", norm2( A @ x_CG[2] - b) )