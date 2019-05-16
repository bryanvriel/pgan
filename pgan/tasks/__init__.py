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


# end of file
