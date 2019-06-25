#-*- coding: utf-8 -*-
  
import pyre
import pgan

@pyre.foundry(implements=pgan.action, tip='Train hidden physics model')
def trainhpm():
    from .TrainHPM import TrainHPM
    return TrainHPM

@pyre.foundry(implements=pgan.action, tip='Train generative physics model')
def traingan():
    from .TrainGAN import TrainGAN
    return TrainGAN

@pyre.foundry(implements=pgan.action, tip='Train variational autoencoder with physics')
def trainvae():
    from .TrainVAE import TrainVAE
    return TrainVAE

@pyre.foundry(implements=pgan.action, tip='Train variational autoencoder with physics')
def trainff():
    from .TrainFF import TrainFF
    return TrainFF

@pyre.foundry(implements=pgan.action, tip='Train physics-informed neural network')
def trainpinn():
    from .TrainPINN import TrainPINN
    return TrainPINN

# end of file
