#-*- coding: utf-8 -*-

import numpy as np
import h5py
from pgan.data import Data

# The bounds
UMIN, UMAX = -0.35, 0.35
VMIN, VMAX = -0.35, 0.35
WMIN, WMAX = -2.0, 2.0
TMIN, TMAX = 0.0, 200.0

def unpack(filename, train_fraction=0.9, batch_likelihood=8, batch_pde=8, norm=False):
    """
    Unpack HDF5 arrays and pack into DataTuple instances.
    """
    # Set the normalization function
    if norm:
        normfun = normalize
    else:
        normfun = identity

    # Create a random number generator
    rng = np.random.RandomState(seed=13)
    
    # Load data
    with h5py.File('data_128x128.h5', 'r') as fid:

        # Get shape information
        Nt, Ny, Nx, Nc = fid['W'].shape

        # Load data
        Tdat = fid['T'][()]
        Wdat = fid['W'][()]
        Udat = fid['U'][()]
        Vdat = fid['V'][()]

        # Randomize initial snapshots and select half for data training
        N_init = 600
        rand_ind = rng.permutation(N_init)
        ind = rand_ind[:N_init//2]
        T0 = Tdat[ind]
        W0 = Wdat[ind]

        # Make a data object
        data = Data(train_fraction=0.85,
                    batch_size=batch_likelihood,
                    shuffle=True,
                    T=normfun(T0, TMIN, TMAX),
                    W=normfun(W0, WMIN, WMAX))

        # Get rest of initial snapshots for physics consistency
        ind = rand_ind[N_init//2:]
        Tp = Tdat[ind]
        Wp = Wdat[ind]
        Up = Udat[ind]
        Vp = Vdat[ind]

        # Use rest of data for physics consistency
        Tp = np.vstack((Tp, Tdat[N_init:]))
        Wp = np.vstack((Wp, Wdat[N_init:]))
        Up = np.vstack((Up, Udat[N_init:]))
        Vp = np.vstack((Vp, Vdat[N_init:]))

        # Make data object for physics consistency
        data_pde = Data(train_fraction=0.85,
                        batch_size=batch_pde,
                        shuffle=True,
                        T=normfun(Tp, TMIN, TMAX),
                        W=normfun(Wp, WMIN, WMAX),
                        U=normfun(Up, UMIN, UMAX),
                        V=normfun(Vp, VMIN, VMAX))

    # The coordinates
    x_vals = np.linspace(0.0, 2.0*np.pi, Nx)
    y_vals = np.linspace(0.0, 2.0*np.pi, Ny)
    x_bounds = [0.0, 2.0*np.pi]
    y_bounds = [0.0, 2.0*np.pi]

    # Normalize
    x_vals = normfun(x_vals, 0.0, 2.0*np.pi)
    y_vals = normfun(y_vals, 0.0, 2.0*np.pi)
    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    # Tack on shape information
    data.ny, data.nx = Ny, Nx

    # Tack on bounds information to data PDE
    data_pde.dx = dx
    data_pde.dy = dy
    data_pde.x_bounds = x_bounds
    data_pde.y_bounds = y_bounds
    data_pde.w_bounds = [WMIN, WMAX]
    data_pde.u_bounds = [UMIN, UMAX]
    data_pde.v_bounds = [VMIN, VMAX]
    data_pde.t_bounds = [TMIN, TMAX]

    # Done
    return data, data_pde

def normalize(x, xmin, xmax):
    """
    Normalize data to be roughly in the range of [-1, 1].
    """
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def identity(x, xmin, xmax):
    """
    Do nothing. Simply return input.
    """
    return x
    
# end of file
