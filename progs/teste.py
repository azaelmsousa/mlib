import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from src import nn

df_data = pd.read_csv("../data/fashion-mnist_test.csv")
data = df_data.values
print(df_data.head(10))

hl1 = nn.layer(10,data.shape[1],'relu','Hidden Layer 01')
hl2 = nn.layer(20,10,'relu','Hidden Layer 02')
lo = nn.layer(2,20,'softmax','output')

neural_network = nn.nn()
neural_network.add(hl1)
neural_network.add(hl2)
neural_network.add(lo)

neural_network.showArchitecture()