import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from utils.utils import normalize_points

from pylab import rcParams
rcParams['figure.figsize'] = 5,5

PATH ="/Volumes/fabioexternal/Dropbox/Apps/masterthesis_results/singulation12_latent1032_mse_posvel_unknown.json/test_20_rollouts_78001_iterations_trained/summary_images_batch_78001_exp_id_8045/obj_pos_vel_dataframe.pkl"

df = pd.read_pickle(PATH)

coord_list_gt = normalize_points(df.ix[1:, 0].tolist())
# for now handle pred separately (normalization error --> divide by 240)
coord_list_pred_scaled = [i/240 for i in df.ix[1:, 1].tolist()]
coord_list_pred = normalize_points(coord_list_pred_scaled)
gt_x = [i[0] for i in coord_list_gt]
gt_y = [i[1] for i in coord_list_gt]
gt_z = [i[2] for i in coord_list_gt]
pred_x = [i[0] for i in coord_list_pred]
pred_y = [i[1] for i in coord_list_pred]
pred_z = [i[2] for i in coord_list_pred]


colors = cm.rainbow(np.linspace(0, 1, len(gt_x)))

""" 3D """
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(gt_x, gt_y, gt_z, marker='o', c=colors, label='gt')
ax.scatter(pred_x, pred_y, pred_z, marker='x', c=colors, label='predicted')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

""" 2D """
#plt.scatter(x=gt_2d_x, y=gt_2d_y, marker='o', s=80, c=colors, label='gt')
#plt.scatter(x=pred_2d_x, y=pred_2d_y, marker='x', s=80, c=colors, label='predicted')

#plt.xlim(0,0.6)
#plt.ylim(0.05, 0.20)
plt.xlim(0,1)
plt.ylim(0,1)
plt.zlim(0,1)
plt.legend()
plt.show()