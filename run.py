# files withing the project
import constants as const
import RandomSample as sampler
import AdjacencyConfusion as AdjConf
import ArrowConfusion as ArrConf

# useful python libraries
import numpy as np
import pandas as pd

# causal-learn package
import causallearn as cl

from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from causallearn.graph.Graph import Graph

import os
import sys
import io
from causallearn.score.LocalScoreFunction import local_score_BDeu
from causallearn.utils.GraphUtils import GraphUtils
sys.path.append("")
import unittest
import warnings
from pickle import load
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np



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

    os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz\bin'

    testpc.draw_pydot_graph()
    

    #Record = ges(X, 'local_score_marginal_general', maxP=maxP, parameters=parameters)
    """
    pyd = GraphUtils.to_pydot(Record)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

    print(Record)

    """
  



    print("finish")



if __name__ == "__main__":
    run()