import constants as const
import RandomSample as sampler

import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz

def run():

    data = sampler.sample(const.p, const.d, const.N)
    #print(data)

    np.savetxt('randomDataset.txt', data, delimiter ='\t')


    testpc = pc(data, const.d, gsq, True, 0, -1 )
    print("---------------------------------------")

    testfci = fci(data, fisherz, const.d, verbose=False)
    print("---------------------------------------")
    
    c_indx = np.reshape(np.asarray(list(range(data.shape[0]))), (data.shape[0],1))
    #print(c_indx)
    testcdnod = cdnod(data, c_indx, const.d, kci, True, 0, -1)
    print("---------------------------------------")

    print("finish")



if __name__ == "__main__":
    run()