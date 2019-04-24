# # todo: remove
# print("=== WARNING: shuffling within single trajectories is activated! (TRAIN) ===")
# s = np.arange(self.config.n_rollouts)
# np.random.shuffle(s)
#
# # shift last rollout to last
# idx_max = np.argmax(s)
# tmp = s[idx_max]
# s[idx_max] = s[-1]
# s[-1] = tmp
#
# for feature in features:
#     feature['img'] = feature['img'][s]
#     feature['seg'] = feature['seg'][s]
#     feature['depth'] = feature['depth'][s]
#     for s_i in s:
#         if s_i + 1 >= self.config.n_rollouts:
#             continue
#         idx_of_next_of_s_i = np.where(s == (s_i + 1))[0][0]
#         feature['gripperpos'][s_i] = feature['gripperpos'][idx_of_next_of_s_i]
#         feature['grippervel'][s_i] = feature['grippervel'][idx_of_next_of_s_i]
#     feature['objpos'] = feature['objpos'][s]
#     feature['objvel'] = feature['objvel'][s]
#     feature['object_segments'] = feature['object_segments'][s]

#input_graphs_all_exp, target_graphs_all_exp, input_ctrl_graphs_all_exp = create_graphs(config=self.config,
#                                                                                       batch_data=features,
#                                                                                       batch_size=self.config.train_batch_size,
#                                                                                       initial_pos_vel_known=self.config.initial_pos_vel_known
#                                                                                       )

# # todo: remove
# # adjust target graphs so that the loss is computed correctly
# for j, batch_target_graph in enumerate(target_graphs_all_exp):
#     idxs = []
#     for s_i in s:
#         if s_i + 1 >= self.config.n_rollouts:
#             continue
#         idxs.append(np.where(s == (s_i + 1))[0][0])  # returns a tuple also zero-indexed, n_rollouts - 1 is the last element in target graphs list
#     batch = []
#     for i in idxs:
#         if i + 1 != self.config.n_rollouts:
#             batch.append(batch_target_graph[i])
#         else:
#             batch.append(batch_target_graph[-1])
#     target_graphs_all_exp[j] = batch
#     #for target_graph in batch_target_graph:
#     #    print('step: ', target_graph.graph['features'][1])


# print("reversing the gripper behaviour")
# for dct in features:
#    for i in range(self.config.n_rollouts-1):
#        if i < self.config.n_rollouts:
#            diff = dct["gripperpos"][i+1] - dct["gripperpos"][i]
#            dct['gripperpos'][i] = dct["gripperpos"][i] - diff
#    dct['grippervel'] = -1 * dct['grippervel']

# print("gripper zero")
# for dct in features:
#    for i in range(self.config.n_rollouts):
#        dct['gripperpos'][i] = np.ones(np.shape(dct['gripperpos'][i])) * np.random.normal(100000, 500.0, 1)
#    dct['grippervel'] = np.ones(np.shape(dct['grippervel'])) * np.random.normal(100000, 500.0, 1)
# dct['objvel'] = np.ones(np.shape(dct['objvel']))*1000
# dct['objpos'] = np.ones(np.shape(dct['objpos']))*1000