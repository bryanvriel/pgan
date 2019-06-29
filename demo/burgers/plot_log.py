#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from tqdm import tqdm
import argparse
import sys

def parse_command_line():
    parser = argparse.ArgumentParser(description="""
        Plot training lots.""")
    parser.add_argument('logfile', type=str, help='Input log file.')
    parser.add_argument('-head', action='store', type=int, default=0,
        help='Number of lines to skip in beginning of log. Default: 0.')
    return parser.parse_args()


def main(args):

    # Load log
    epochs, losses = read_log(args.logfile)
    nvar = losses.shape[1] // 2

    # Skip some lines
    epochs = epochs[args.head:]
    losses = losses[args.head:]

    # Median filter
    for j in tqdm(range(losses.shape[1])):
        losses[:,j] = medfilt(losses[:,j], kernel_size=51)

    # Decimate
    epochs = epochs[::25]
    losses = losses[::25]

    # Labels
    labels = ['disc', 'gen', 'MSE', 'PDE']

    fig, axes = plt.subplots(nrows=nvar, figsize=(8,7))
    for i in range(nvar):
        axes[i].plot(epochs, losses[:,i], alpha=0.7, label='Train')
        axes[i].plot(epochs, losses[:,i+nvar], alpha=0.7, label='Test')
        axes[i].set_ylabel(labels[i])
        leg = axes[i].legend(loc='best')
        axes[i].grid(True, linestyle=':')

    fig.set_tight_layout(True)
    plt.show()


def read_log(filename):
    with open(filename, 'r') as fid:
        epochs = []
        losses = []
        for input_line in fid:
            if not input_line.startswith('INFO:root'):
                continue
            fields = input_line.strip().split(':')
            dat = fields[-1].split()
            epoch = int(dat[0])
            loss = [float(x) for x in dat[1:]]
            epochs.append(epoch)
            losses.append(loss)

    return np.array(epochs), np.array(losses)


if __name__ == '__main__':
    args = parse_command_line()
    main(args)

# end of file
