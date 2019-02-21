import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import os
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import numpy as np
from utils.utils import normalize_points

from pylab import rcParams
rcParams['figure.figsize'] = 5,5

PATH ="/Volumes/fabioexternal/Dropbox/Apps/masterthesis_results/test_10_rollouts_78001_iterations_trained/test_10_rollouts_78001_iterations_trained/summary_images_batch_78001_exp_id_6577/obj_pos_vel_dataframe.pkl"
SELECTED_OBJECT_NUMBER = 1 # 0, 1, 2

df = pd.read_pickle(PATH)

#coord_list_gt = df.ix[1:, SELECTED_OBJECT_NUMBER*2].tolist()
coord_list_gt = normalize_points(df.ix[1:, SELECTED_OBJECT_NUMBER*2].tolist())
# for now handle pred separately (normalization error --> divide by 240)
coord_list_pred_scaled = [i/240 for i in df.ix[1:, (SELECTED_OBJECT_NUMBER*2)+1].tolist()]
#coord_list_pred = coord_list_pred_scaled
coord_list_pred = normalize_points(coord_list_pred_scaled)





""" 3D """
gt_x = [i[0] for i in coord_list_gt]
gt_y = [i[1] for i in coord_list_gt]
gt_z = [i[2] for i in coord_list_gt]
pred_x = [i[0] for i in coord_list_pred]
pred_y = [i[1] for i in coord_list_pred]
pred_z = [i[2] for i in coord_list_pred]
colors = cm.rainbow(np.linspace(0, 1, len(gt_x)))

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax = fig.gca(projection='3d')
#ax = Axes3D(fig)

imgs = []
#fig = plt.figure(frameon=False)
#ax = fig.add_subplot(111, projection='3d')
fig = plt.figure()
#ax = plt.Axes(fig, [0.,0.,1.,1.])
ax = fig.add_subplot(111, projection='3d')
for i in range(len(gt_x)):
    ax.scatter(gt_x[i], gt_y[i], gt_z[i], s=50, marker='o', c=colors[i], label='gt')
    ax.scatter(pred_x[i], pred_y[i], pred_z[i], s=50, marker='x', c=colors[i], label='predicted')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.box(on=None)
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    #plt.legend()
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imgs.append([plt.imshow(data, animated=True)])

clip = animation.ArtistAnimation(fig, imgs, interval=100, blit=False, repeat_delay=1000)
name = os.path.join("../data", "scatter_plot") + ".gif"
clip.save(name, writer='imagemagick')


# ax.scatter(gt_x, gt_y, gt_z, s=50, marker='o', c=colors, label='gt')
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
# ax.set_zlim(0,1)
# ax.view_init(elev=10, azim=20)
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
#
# #ax.set_xlim(0.344, 0.856)
# #ax.set_ylim(-0.256, 0.256)
# #ax.set_zlim(-0.149,-0.0307)
# plt.legend()
#plt.show(block=True)


# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.1)

""" 2D """
#plt.scatter(x=gt_x[:,2], y=gt_2d_y, marker='o', s=80, c=colors, label='gt')
#plt.scatter(x=pred_2d_x, y=pred_2d_y, marker='x', s=80, c=colors, label='predicted')

#plt.xlim(0,0.6)
#plt.ylim(0.05, 0.20)
#plt.xlim(0,1)
#plt.ylim(0,1)
#plt.zlim(0,1)
#plt.legend()
#plt.show()