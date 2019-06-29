#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, filters
from scipy.io import loadmat
import h5py
import sys
import os

def main():

    # Load original coordinates
    with h5py.File('data_full.h5', 'r') as fid:
        x = fid['x_coords'][()]
        t = fid['t_coords'][()]
    Nx = len(x)
    Nt = len(t)

    # Load predictions
    with h5py.File('output_predictions.h5', 'r') as fid:
        mean = fid['mean'][()].reshape(Nx, Nt)
        std = fid['std'][()].reshape(Nx, Nt)
        U_ref = fid['ref'][()].reshape(Nx, Nt)
    print('MSE:', np.mean((U_ref - mean)**2))

    # Check derivatives
    dx = np.mean(np.diff(x))
    dt = np.mean(np.diff(t))

    u = U_ref
    u_x, u_t = np.gradient(u, dx, dt, axis=(0, 1))
    u_xx = np.gradient(u_x, dx, axis=0)
    f_ref = u_t + u * u_x - 0.1 * u_xx
    print(np.mean(f_ref**2))

    u = mean
    u_x, u_t = np.gradient(u, dx, dt, axis=(0, 1))
    u_xx = np.gradient(u_x, dx, axis=0)
    f_mean_vals = u_t + u * u_x - 0.1 * u_xx
    print(np.mean(f_mean_vals**2))

    # Make figure
    extent = (t[0], t[-1], x[-1], x[0])
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, figsize=(8,8.5))
    im1 = ax1.imshow(mean, aspect='auto', cmap='jet', extent=extent)
    im2 = ax2.imshow(U_ref, aspect='auto', cmap='jet', extent=extent)
    im3 = ax3.imshow(f_ref, aspect='auto', extent=extent)
    im4 = ax4.imshow(f_mean_vals, aspect='auto', extent=extent)
    im5 = ax5.imshow(std, aspect='auto', cmap='jet', extent=extent)
    im1.set_clim(im2.get_clim())
    im4.set_clim(im3.get_clim())
    im5.set_clim(0, 0.005)

    # Colorbars
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar4 = plt.colorbar(im4, ax=ax4)
    cbar5 = plt.colorbar(im5, ax=ax5)

    # Colorbar labels
    cbar1.set_label('Generated mean')
    cbar2.set_label('True')
    cbar3.set_label('PDE loss true')
    cbar4.set_label('PDE loss generated')
    cbar5.set_label('Generated st. dev.')

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_ylabel('X')
        ax.set_xticklabels([])
    ax5.set_ylabel('X')
    ax5.set_xlabel('T')

    fig.set_tight_layout(True)
    plt.show()    

if __name__ == '__main__':
    main()

# end of file
