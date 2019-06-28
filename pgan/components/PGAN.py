#-*- coding: utf-8 -*-

import pyre

class PGAN(pyre.plexus, family='pgan.global'):
    """
    The pgan executive and application wrapper.
    """

    pyre_namespace = 'pgan'
    from .Action import Action as pyre_action

    inter_op_cores = pyre.properties.int(default=1)
    inter_op_cores.doc = 'Number of cores for inter_op_parallelism'

    intra_op_threads = pyre.properties.int(default=1)
    intra_op_threads.doc = 'Number of threads for intra_op_parallelism'

    batch_size = pyre.properties.int(default=128)
    batch_size.doc = 'Batch size'
    
    def main(self, *args, **kwargs):
        """
        The main entrypoint into the plexus.
        """
        super().main(*args, **kwargs)

    def help(self, **kwargs):
        """
        Embellish the pyre.plexus help by printing out the banner first.
        """
        # Get pgan package
        import pgan
        # Get a channel
        channel = self.info
        # Make some space
        channel.line()
        # Print header
        channel.line(pgan.meta.header)
        # Call plexus help
        super().help(**kwargs)


# end of file 
