from fluxDataset import *
import numpy as np

fd1 = FluxDataset("./data/samples/small_human", 300, 0, seed=0)
fd2 = FluxDataset("./data/samples/small_human", 300, 0, seed=0)

print(np.max(fd1.values - fd2.values))