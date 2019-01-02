import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from src import nn

'''
df_data = pd.read_csv("../data/fashion-mnist_test.csv")
print(df_data.head(10))
data = df_data.values
data = data / 255

hl1 = nn.layer(20,data.shape[1],'ReLU','Hidden Layer 01')
hl2 = nn.layer(10,20,'ReLU','Hidden Layer 02')
lo = nn.layer(10,10,'softmax','output')

ann = nn.nn()
ann.add(hl1)
ann.add(hl2)
ann.add(lo)
ann.loadInput(data)

classes = ann.forward()
print(classes)
'''

#------------------------------------------
#               Toy Example
#------------------------------------------

'''
data = np.array([[0.05,0.1]])

w1 = [np.array([0.35,0.15,0.20]),np.array([0.35,0.25,0.30])]
hl1 = nn.layer(2,2,'sigmoid','Hidden Layer 01',weights=w1)

w2 = [np.array([0.60,0.40,0.45]),np.array([0.60,0.50,0.55])]
lo = nn.layer(2,2,'sigmoid','Output Layer',weights=w2)

ann = nn.nn()
ann.add(hl1)
ann.add(lo)
ann.loadInput(data)

classes = ann.forward()
print(classes)
'''
