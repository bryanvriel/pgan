#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
import sys
import os

import pgan
import pyre

def parse_command_line():
    parser = argparse.ArgumentParser(description='Generate stochastic predictions.')
    parser.add_argument('checkdir', type=str, help='Checkpoints directory to load from.')
    parser.add_argument('datafile', type=str, help='HDF5 data file to read.')
    return parser.parse_args()


def main(args):

    # Load the pfg configuration
    pfg = load_configuration('pgan.pfg') 

    # Load data objects from full data file
    data_gan, data_pde = pfg.module.unpack(args.datafile, train_fraction=1.0, shuffle=False)

    # Separately create the PDE network
    if pfg.use_known_pde:
        pde_net = pfg.module.KnownPDENet()
    else:
        pde_net = pfg.module.PDENet(pfg.pde_layers)

    # Create the GAN model
    model = pfg.module.GAN(
        generator_layers=pfg.generator_layers,
        discriminator_layers=pfg.discriminator_layers,
        latent_dims=1,
        physical_model=pde_net,
        pde_beta=1.0
    )

    # Construct graphs
    model.build(inter_op_cores=0,
                intra_op_threads=0)
    model.print_variables()

    # Load PDE weights separately
    if not pfg.use_known_pde:
        saver = tf.train.Saver(var_list=pde_net.trainable_variables)
        saver.restore(model.sess, os.path.join(pfg.pde_checkdir, 'pde.ckpt'))

    # Load previous checkpoints
    model.load(indir=args.checkdir)

    # Generate predictions
    batch = data_gan.train
    U_samples = model.predict(batch['X'], batch['T'], n_samples=1000)

    # Compute stats
    mean = np.mean(U_samples, axis=0)
    std = np.std(U_samples, axis=0)

    # Save to file
    with h5py.File('output_predictions.h5', 'w') as fid:
        fid['mean'] = mean
        fid['std'] = std
        fid['ref'] = batch['U']


def load_configuration(filename, section='traingan'):

    # Instantiate a dummy class for storing parameters
    pfg = GenericClass()

    # Create a nameserver and load the pfg file
    ns = pyre.executive.nameserver
    pyre.loadConfiguration(filename)

    # The data file
    pfg.data_file = ns['pgan.traingan.data_file']
    
    # Get dynamics submodule from pfg file
    pfg.module = getattr(pgan.dynamics, ns['pgan.traingan.dynamics'])

    # Load layer sizes from pfg file
    pfg.generator_layers = [
        int(a) for a in ns['pgan.traingan.generator_layers'].split(',')
    ]
    pfg.discriminator_layers = [
        int(a) for a in ns['pgan.traingan.discriminator_layers'].split(',')
    ]

    # Physics parameters
    pfg.use_known_pde = bool(ns['pgan.traingan.use_known_pde'])
    if not pfg.use_known_pde:
        pfg.pde_layers = [
            int(a) for a in ns['pgan.traingan.pde_layers'].split(',')
        ]
        pfg.pde_checkdir = ns['pgan.traingan.pde_checkdir']

    return pfg


class GenericClass:
    pass


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
