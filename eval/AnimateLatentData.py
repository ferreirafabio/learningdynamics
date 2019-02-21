import matplotlib
matplotlib.use('Agg') # in terminal change to 'Agg'
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.markers as mmarkers
import numpy as np
from utils.math_ops import normalize_df_column
import pandas as pd


class AnimateLatentData():
    def __init__(self, df, identifier1, identifier2):
        #self.df = pd.read_pickle(path_to_df)
        self.df = df
        self.id1 = identifier1
        self.id2 = identifier2
        # for now handle pred separately (normalization error --> divide by 240)
        self.df[self.id2] = self.df[self.id2]/240

        # cut the first value due to random initialization
        self.pos_gt = normalize_df_column(self.df, self.id1)[1:].to_frame()
        self.pos_pred = normalize_df_column(self.df, self.id2)[1:].to_frame()
        self.fig = plt.figure(dpi=200)

    def store_3dplot(self, title, output_dir):
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.graph = self.ax.scatter([], [], [], s=70, marker="o")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_zlim(0, 1)
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.graph.set_alpha(1)
        self.colors = cm.tab10(np.linspace(0, 1, len(self.pos_gt)))

        self.pos_gt.loc[:, 'x'] = self.pos_gt[self.id1].apply(lambda row: row[0])
        self.pos_gt.loc[:, 'y'] = self.pos_gt[self.id1].apply(lambda row: row[1])
        self.pos_gt.loc[:, 'z'] = self.pos_gt[self.id1].apply(lambda row: row[2])

        self.pos_pred.loc[:, 'x'] = self.pos_pred[self.id2].apply(lambda row: row[0])
        self.pos_pred.loc[:, 'y'] = self.pos_pred[self.id2].apply(lambda row: row[1])
        self.pos_pred.loc[:, 'z'] = self.pos_pred[self.id2].apply(lambda row: row[2])

        self.ax.set_title(title)

        legend_handle_gt = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5)
        legend_handle_pred = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=5)
        self.ax.legend(loc='lower left', handles=[legend_handle_gt, legend_handle_pred], labels=["ground truth", "predicted"])
        self.ax.view_init(70, 210)

        self.ani = matplotlib.animation.FuncAnimation(self.fig, self._update_3d_graph, frames=9, interval=600, repeat_delay=5000, blit=False)
        self.ani.save(output_dir, writer="imagemagick")


    def store_2dplot(self, title, output_dir):
        self.fig = plt.figure(dpi=200)
        self.ax = self.fig.add_subplot(111)
        self.graph = self.ax.scatter([], [], s=70, marker="o", animated=True)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        plt.gca().invert_yaxis()
        #plt.gca().invert_xaxis()
        self.ax.set_xlabel("y")
        self.ax.set_ylabel("x")
        self.graph.set_alpha(1)
        self.colors = cm.tab10(np.linspace(0, 1, len(self.pos_gt)))

        self.pos_gt.loc[:, 'x'] = self.pos_gt[self.id1].apply(lambda row: row[0])
        self.pos_gt.loc[:, 'y'] = self.pos_gt[self.id1].apply(lambda row: row[1])

        self.pos_pred.loc[:, 'x'] = self.pos_pred[self.id2].apply(lambda row: row[0])
        self.pos_pred.loc[:, 'y'] = self.pos_pred[self.id2].apply(lambda row: row[1])

        self.ax.set_title(title)
        legend_handle_gt = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5)
        legend_handle_pred = mlines.Line2D([], [], color='black', marker='x', linestyle='None', markersize=5)
        self.ax.legend(loc='lower left', handles=[legend_handle_gt, legend_handle_pred], labels=["ground truth", "predicted"])


        self.ani = matplotlib.animation.FuncAnimation(self.fig, self._update_2d_graph, frames=9, interval=600, repeat_delay=5000, blit=False)
        self.ani.save(output_dir, writer="imagemagick")

    def _update_3d_graph(self, num):
        x_updated = np.concatenate([self.pos_gt.x[:num+1].tolist(), self.pos_pred.x[:num+1].tolist()])
        y_updated = np.concatenate([self.pos_gt.y[:num+1].tolist(), self.pos_pred.y[:num+1].tolist()])
        z_updated = np.concatenate([self.pos_gt.z[:num+1].tolist(), self.pos_pred.z[:num+1].tolist()])

        paths_gt = []
        paths_ped = []

        for i in range(num+1):
            marker_gt = mmarkers.MarkerStyle('o')
            marker_ped = mmarkers.MarkerStyle('x')
            path_gt = marker_gt.get_path().transformed(marker_gt.get_transform())
            path_ped = marker_ped.get_path().transformed(marker_ped.get_transform())
            paths_gt.append(path_gt)
            paths_ped.append(path_ped)

        self.graph._paths = np.concatenate([paths_gt, paths_ped])
        self.graph._facecolor3d = np.concatenate([self.colors[:num+1], self.colors[:num+1]])
        self.graph._edgecolor3d = np.concatenate([self.colors[:num+1], self.colors[:num+1]])
        self.graph._offsets3d = (x_updated, y_updated, z_updated)

    def _update_2d_graph(self, num):
        x_updated = np.hstack((self.pos_gt.x[:num + 1].tolist(), self.pos_pred.x[:num + 1].tolist()))
        y_updated = np.hstack((self.pos_gt.y[:num + 1].tolist(), self.pos_pred.y[:num + 1].tolist()))

        paths_gt = []
        paths_ped = []

        for i in range(num + 1):
            marker_gt = mmarkers.MarkerStyle('o')
            marker_ped = mmarkers.MarkerStyle('x')
            path_gt = marker_gt.get_path().transformed(marker_gt.get_transform())
            path_ped = marker_ped.get_path().transformed(marker_ped.get_transform())
            paths_gt.append(path_gt)
            paths_ped.append(path_ped)

        self.graph._paths = np.concatenate([paths_gt, paths_ped])
        self.graph.set_facecolors(np.concatenate([self.colors[:num + 1], self.colors[:num + 1]]))
        self.graph.set_edgecolors(np.concatenate([self.colors[:num + 1], self.colors[:num + 1]]))
        self.graph.set_offsets(np.hstack((y_updated[:, np.newaxis], x_updated[:, np.newaxis])))







if __name__ == '__main__':
    PATH = "/Volumes/fabioexternal/Dropbox/Apps/masterthesis_results/test_10_rollouts_78001_iterations_trained/test_10_rollouts_78001_iterations_trained/summary_images_batch_78001_exp_id_6577/obj_pos_vel_dataframe.pkl"
    SELECTED_OBJECT_NUMBER = 0  # 0, 1, 2
    identifier_gt = "{}_obj_gt_pos".format(SELECTED_OBJECT_NUMBER)
    identifier_pred = "{}_obj_pred_pos".format(SELECTED_OBJECT_NUMBER)

    out_dir = "../data/animation.gif"
    animate = AnimateLatentData(path_to_df=PATH, identifier1=identifier_gt, identifier2=identifier_pred)
    title = 'Ground truth vs predicted centroid position of object {}'.format(SELECTED_OBJECT_NUMBER)
    animate.store_3dplot(title=title, output_dir=out_dir)
    animate.store_2dplot(title=title, output_dir=out_dir+"2.gif")

