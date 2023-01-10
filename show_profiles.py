import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

with open('prof.pkl', 'rb') as f:
    profiles = pkl.load(f)

for comp in profiles.keys():
    plt.plot(profiles[comp])

plt.show()