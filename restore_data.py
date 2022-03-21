import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob


data_folders = glob.glob('data_*')
# print(data_folders)

with open(f'{data_folders[0]}/parameters.pkl', 'rb') as f:
    parameters_data = pkl.load(f)

with open(f'{data_folders[0]}/profiles.pkl', 'rb') as f:
    profiles_data = pkl.load(f)
