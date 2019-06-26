import argparse

import pickle as pkl
import scipy as sc
import matplotlib.pyplot as plt
import tools.pseudospectral as nst

from scipy.fftpack import fft2, ifft2


def run(args):
    """Deals with simulatiing some fluid dynamics in 2D with periodic boundary 
    conditions.

    """
    
    # Physical constants
    nu = 1/args.reynolds
    
    # Size of the simulations domain
    lx = 2*sc.pi
    ly = 2*sc.pi
    
    # Number of grid points
    nx = args.resolution
    ny = args.resolution
    
    # Grid increments
    dx = lx/nx
    dy = ly/ny
    
    # Initialize vorticity field
    omega, p = nst.dancing_vortices(nx, ny, dx, dy)
    
    # Gradient operators in Fourier domain for x- and y-direction
    Kx, Ky = nst.spectral_gradient(nx, ny, lx, ly)
    
    # 2D Laplace operator and 2D inverse Laplace operator in Fourier domain
    K2, K2inv = nst.spectral_laplace(nx, ny, Kx, Ky)
    
    # Simulation time
    t_end = args.t_end
    
    # Set discrete time step by choosing CFL number (condition: CFL <= 1)
    CFL = 1
    u = sc.real(ifft2(-Ky*K2inv*fft2(omega)))
    v = sc.real(ifft2(Kx*K2inv*fft2(omega)))
    u_max = sc.amax(sc.absolute(u))
    v_max = sc.amax(sc.absolute(v))
    t_step = (CFL*dx*dy)/(u_max*dy+v_max*dx)
    
    # Start Simulation
    t_sum = 0
    i = 0
    history = [(t_sum, omega)]
    while t_sum <= t_end:
        # Runge-Kitta 4 time simulation
        omega = nst.rk4(t_step, omega, Kx, Ky, K2, K2inv, nu)
    
        # Plot every 100th frame
        if 0 == i % 100:
    
            plt.imshow(omega)
            plt.pause(0.1)
    
        i += 1
        t_sum += t_step
        
        # Store vorticity field at this time
        history.append((t_sum, omega))
        print("Step {} at time {}.".format(i, t_sum))

    # Store simulation vorticity field
    with open(args.output, 'wb') as fid:
        pkl.dump(history, fid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate some fluid dynamics.')
    parser.add_argument('--reynolds', type=float, default=1000, help='Reynolds number')
    parser.add_argument('--t_end', type=float, default=1.0, help='Simulation time')
    parser.add_argument('--resolution', type=int, default=512, help='Number of grid points')
    parser.add_argument('--output', type=str, help='Output filepath')

    args = parser.parse_args()
    print(args)
    
    # Run the simulation
    run(args)

