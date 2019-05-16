#-*- coding: utf-8 -*-

import numpy as np
import pyDOE

def train_test_boundary(x, t, u, seed=42, n_initial=100, n_boundary=50, n_collocation=10000,
                        train_fraction=0.8, shuffle=True):

    # Set the seed
    np.random.seed(seed)

    # Sample initial condition
    u0 = u[0].ravel()
    inds = np.random.choice(u0.size, size=n_initial, replace=False)
    X = x[0].ravel()[inds]
    T = np.full(n_initial, t[0])
    U = u0[inds]

    # Sample boundary conditions
    if u.ndim == 2: # (1 time, 1 spatial)

        # Lower boundary
        u_lb = u[:,0]
        inds = np.random.choice(u_lb.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[0])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))

        # Upper boundary
        u_ub = u[:,-1]
        inds = np.random.choice(u_ub.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[-1])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))

    elif u.ndim == 3: # (1 time, 2 spatial)

        # Lower boundary 1
        u_lb = u[:,0,:].ravel()
        inds = np.random.choice(u_lb.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[0])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))

        # Lower boundary 2
        u_lb = u[:,:,0].ravel()
        inds = np.random.choice(u_lb.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[0])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))
        
        # Upper boundary 1
        u_ub = u[:,-1,:].ravel()
        inds = np.random.choice(u_ub.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[-1])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))

        # Upper boundary 2
        u_ub = u[:,:,-1].ravel()
        inds = np.random.choice(u_ub.size, size=n_boundary, replace=False)
        X = np.hstack((X, np.full(n_boundary, x[-1])))
        T = np.hstack((T, t[inds]))
        U = np.hstack((U, u_lb[inds]))

    else:
        raise NotImplementedError

    # Sample collocation points
    if u.ndim == 2:
        lower = np.array([t[0], x[0]])
        upper = np.array([t[-1], x[-1]])
    elif u.ndim == 3:
        lower = np.array([t[0], x[0,0], x[0,0]])
        upper = np.array([t[-1], x[-1,0], x[0,-1]])
    else:
        raise NotImplementedError
    Z = lower + (upper - lower) * pyDOE.lhs(len(lower), n_collocation)
    X_coll = Z[:,1:]
    T_coll = Z[:,0]

    # Partition into training and testing
    n_train = int(train_fraction * X.size)
    if shuffle:
        ind = np.random.permutation(X.size)
    else:
        ind = np.arange(X.size, dtype=int)
    X_train = X[ind][:n_train]
    T_train = T[ind][:n_train]
    U_train = U[ind][:n_train]
    X_test = X[ind][n_train:]
    T_test = T[ind][n_train:]
    U_test = U[ind][n_train:]

    # Repeat for collocation points
    


