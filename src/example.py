#!/usr/bin/env python
import numpy as np
from Astar_lib import *
working_matrix=np.load('resources/example_map.npy').T
x1, y1=130, 140
x2, y2= 10, 140
working_matrix[working_matrix > 0]=1 
path=astar(working_matrix, (x2,y2), (x1,y1))
show_path(working_matrix, path) 
