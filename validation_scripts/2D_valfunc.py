# Enable import from parent package
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import dataio, utils, training, loss_functions, modules

import torch
import numpy as np
import math
from torch.utils.data import DataLoader


ckpt_path = './Deepreach_trained_checkpoints/2D_ckpt.pth'
logging_root = './logs'

model = modules.SingleBVPNet(in_features=3, out_features=1, type='sine', mode='mlp',
                             final_layer_factor=1., hidden_features=512, num_hidden_layers=3)
model.cuda()
checkpoint = torch.load(ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
try:
  model_weights = checkpoint['model']
except:
  model_weights = checkpoint
model.load_state_dict(model_weights)
model.eval()


# Time values at which the function needs to be plotted
times = [0., 0.5, 1.0]
num_times = len(times)

# Create a figure
fig = plt.figure(figsize=(5*num_times, 5))

# Get the meshgrid in the (x, y) coordinate
sidelen = 200
mgrid_coords = dataio.get_mgrid(sidelen)

# Start plotting the results
for i in range(num_times):
  time_coords = torch.ones(mgrid_coords.shape[0], 1) * times[i]

  coords = torch.cat((time_coords, mgrid_coords), dim=1) 
  model_in = {'coords': coords.cuda()}
  model_out = model(model_in)['model_out']

  # Detatch model ouput and reshape
  model_out = model_out.detach().cpu().numpy()
  model_out = model_out.reshape((sidelen, sidelen)) 

  # Plot the zero level sets
  model_out = (model_out <= 0.001)*1.

  # Plot the actual data
  ax = fig.add_subplot(1, num_times, 1 + i)
  ax.set_title('t = %0.2f' % times[i])
  s = ax.imshow(model_out.T, cmap='bwr', origin='lower', extent=(-1., 1., -1., 1.))
  fig.colorbar(s) 

fig.savefig(os.path.join(logging_root, '2D_value_function.png'))
