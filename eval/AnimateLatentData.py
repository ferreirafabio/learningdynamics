import matplotlib
matplotlib.use('Agg') # in terminal change to 'Agg', 'TkAgg' on desktop
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.markers as mmarkers
import numpy as np
from utils.math_ops import normalize_df_column
import pandas as pd
import os

class AnimateLatentData():
    def __init__(self, df, identifier1, identifier2, n_rollouts):
        #self.df = pd.read_pickle(path_to_df)
        self.df = df
        self.id1 = identifier1
        self.id2 = identifier2
        self.n_rollouts = n_rollouts-1

        # cut the first value due to random initialization
        self.pos_gt = normalize_df_column(self.df, self.id1)[1:].to_frame()
        self.pos_pred = normalize_df_column(self.df, self.id2)[1:].to_frame()
        mean_col_name = "mean(" + self.id1 + "-" + self.id2 + ")"
        std_col_name = "std(" + self.id1 + "-" + self.id2 + ")"
        self.mean_stats = self.df[mean_col_name].iloc[0]
        self.std_stats = self.df[std_col_name].iloc[0]


    def store_3dplot(self, title, output_dir):
        self.fig = plt.figure(dpi=200)
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
        self.lines = sum([self.ax.plot([], [], [], '-', c=c) for c in self.colors], []) # todo: remove

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
        self.ax.view_init(20, 240)

        ani = matplotlib.animation.FuncAnimation(self.fig, self._update_3d_graph, frames=self.n_rollouts, interval=1000, repeat_delay=5000, blit=False)
        ani.save(output_dir, writer="imagemagick")

        # store static final image
        base, filename = os.path.split(output_dir)
        file, _ = filename.split(".")
        output_dir = os.path.join(base, file + "_final" + ".png")
        self.fig.savefig(output_dir, writer="imagemagick")

        fig_as_np_array = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_as_np_array = fig_as_np_array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return fig_as_np_array


    def store_2dplot(self, title, output_dir):
        self.fig = plt.figure(dpi=200)
        self.ax = self.fig.add_subplot(111)
        self.graph = self.ax.scatter([], [], s=70, marker="o" , animated=True)
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

        self.ax.text(0.0, 0.0, "mean diff per dim: " + str(self.mean_stats) + "\n" + "stddev per dim: " + str(self.std_stats), fontsize=6)

        self.ani = matplotlib.animation.FuncAnimation(self.fig, self._update_2d_graph, frames=self.n_rollouts, interval=1000, repeat_delay=5000, blit=False)
        self.ani.save(output_dir, writer="imagemagick")

        # store static final image
        base, filename = os.path.split(output_dir)
        file, _ = filename.split(".")
        output_dir = os.path.join(base, file + "_final" + ".png")

        plt.show()
        self.fig.savefig(output_dir, writer="imagemagick")

        fig_as_np_array = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        fig_as_np_array = fig_as_np_array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return fig_as_np_array

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

    df = pd.read_pickle(PATH)
    out_dir = "../data/"
    animate = AnimateLatentData(df=df, identifier1=identifier_gt, identifier2=identifier_pred, n_rollouts=10)
    title = 'Ground truth vs predicted centroid position of object {}'.format(SELECTED_OBJECT_NUMBER)
    path_3d = out_dir + "/3d_obj_pos_3d_object" + ".gif"
    path_2d = out_dir + "/2d_obj_pos_3d_object" + ".gif"
    animate.store_3dplot(title=title, output_dir=path_3d)
    animate.store_2dplot(title=title, output_dir=path_2d)

