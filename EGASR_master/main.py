import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import numpy as np
import random

checkpoint = utility.checkpoint(args)

if __name__ == '__main__':
    if checkpoint.ok:
        # ## random seed
        seed = args.seed
        if seed is None:
            seed = random.randint(1, 10000)
        utility.set_random_seed(seed)
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()
