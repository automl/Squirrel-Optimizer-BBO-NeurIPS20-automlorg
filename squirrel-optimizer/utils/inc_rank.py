# Script use to collect incumbents_v5.pkl
# Uses incumbents_v4.pkl and reorders the list in a semi-deterministic manner

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
# from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull, convex_hull_plot_2d

# Incumbents for 'lasso' from regression tasks
lasso = [{'alpha': 0.0588816078969954,
  'fit_intercept': 0,
  'normalize': 0,
  'max_iter': 1188,
  'tol': 0.0142116958607831,
  'positive': 0},
 {'alpha': 0.01,
  'fit_intercept': 1,
  'normalize': 0,
  'max_iter': 5000,
  'tol': 0.0934421821777098,
  'positive': 0},
 {'alpha': 0.0301443773293404,
  'fit_intercept': 1,
  'normalize': 1,
  'max_iter': 465,
  'tol': 0.0994437776399929,
  'positive': 0},
 {'alpha': 0.0342844214778938,
  'fit_intercept': 1,
  'normalize': 1,
  'max_iter': 20,
  'tol': 0.086836065868564,
  'positive': 0}]

# Incumbents for 'linear' from regression tasks
linear = [{'alpha': 0.0807158611555724,
  'fit_intercept': 1,
  'normalize': 1,
  'max_iter': 1482,
  'tol': 0.000985256913005844},
 {'alpha': 0.0158280830848803,
  'fit_intercept': 1,
  'normalize': 1,
  'max_iter': 2024,
  'tol': 0.000274213922897436},
 {'alpha': 0.0131360370365985,
  'fit_intercept': 1,
  'normalize': 0,
  'max_iter': 27,
  'tol': 0.000758983848008941},
 {'alpha': 0.0286632897398748,
  'fit_intercept': 1,
  'normalize': 0,
  'max_iter': 257,
  'tol': 0.000567925032398133}]


def load_model(model_name):
    with open('submissions/switching-optimizer/utils/incumbents_v4.pkl', 'rb') as f:
        inc = pickle.load(f)

    model_incs = inc.loc[model_name].dropna()

    incs = []
    for index in model_incs: incs.extend(index)
    incs = pd.DataFrame(incs)
    return incs


def plot_LR_hull():
    model = load_model('linearC')  # model name used in v4
    mean_point = 10 ** np.log10(model).mean(axis=0)
    mean_dist = distance_matrix([np.log10(mean_point)], np.log10(model))
    closest_point = model.iloc[mean_dist.argmin()]

    hull = ConvexHull(model)
    x = model.iloc[hull.vertices, 0].to_numpy()
    x = np.append(x, x[0])
    y = model.iloc[hull.vertices, 1].to_numpy()
    y = np.append(y, y[0])

    plt.scatter(model['C'], model['intercept_scaling'])
    plt.plot(x, y, color='red')
    plt.scatter(mean_point['C'], mean_point['intercept_scaling'], label='mean point')
    plt.scatter(closest_point['C'], closest_point['intercept_scaling'], label='closest point to mean')
    plt.legend()
    plt.xscale('log'); plt.yscale('log')
    plt.show()


def plot_LR_hull_and_topN(N=16):

    fig, axes = plt.subplots(2, 2)

    LR = create_order('linearC')
    hull = ConvexHull(LR)
    x = LR.iloc[hull.vertices, 0].to_numpy()
    x = np.append(x, x[0])
    y = LR.iloc[hull.vertices, 1].to_numpy()
    y = np.append(y, y[0])
    mean_point = 10 ** np.log10(LR).mean(axis=0)
    mean_dist = distance_matrix([np.log10(mean_point)], np.log10(LR))
    closest_point = LR.iloc[mean_dist.argmin()]

    # the incumbent space from v4
    for i in range(2):
        for j in range(2):
            axes[i, j].scatter(LR['C'], LR['intercept_scaling'], color='black')
            axes[i, j].plot(x, y, color='black', alpha=0.7)
            axes[i, j].scatter(mean_point['C'], mean_point['intercept_scaling'], label='mean point')
            axes[i, j].scatter(closest_point['C'], closest_point['intercept_scaling'], label='closest point to mean')
            axes[i, j].set_xscale('log'); axes[i, j].set_yscale('log')
    # trial 1
    LR = LR.iloc[:N, :]
    axes[0, 0].scatter(LR['C'], LR['intercept_scaling'], s=80, facecolors='none', edgecolors='red', label='top {}'.format(N))
    axes[i, j].legend()

    # trial 2
    LR = create_order('linearC')
    LR = LR.iloc[:N, :]
    axes[0, 1].scatter(LR['C'], LR['intercept_scaling'], s=80, facecolors='none', edgecolors='red', label='top {}'.format(N))

    # trial 3
    LR = create_order('linearC')
    LR = LR.iloc[:N, :]
    axes[1, 0].scatter(LR['C'], LR['intercept_scaling'], s=80, facecolors='none', edgecolors='red', label='top {}'.format(N))

    # trial 4
    LR = create_order('linearC')
    LR = LR.iloc[:N, :]
    axes[1, 1].scatter(LR['C'], LR['intercept_scaling'], s=80, facecolors='none', edgecolors='red', label='top {}'.format(N))

    for i in range(2):
        for j in range(2):
            axes[i, j].legend()

    plt.suptitle('LR incumbents')
    plt.show()



def create_order(model_name='linearC'):
    if model_name == 'linear':
        return pd.DataFrame(linear)
    if model_name == 'lasso':
        return pd.DataFrame(lasso)

    model = load_model(model_name)
    hull = ConvexHull(model)
    mean_point = 10 ** np.log10(model).mean(axis=0)

    incs_added = []
    hull_vertex_added = []

    # finding point closest to mean
    mean_dist_all = distance_matrix([np.log10(mean_point)], np.log10(model))
    closest_point = model.iloc[mean_dist_all.argmin()]
    incs_added.append(mean_dist_all.argmin())

    # distance between the vertices
    hull_dist = distance_matrix(np.log10(model.iloc[hull.vertices]),
                                np.log10(model.iloc[hull.vertices]))
    # distance of closest point to mean from all vertices
    dist_point_vertices = distance_matrix([np.log10(closest_point)],
                                          np.log10(model.iloc[hull.vertices]))

    # store incumbent list, ranked through a heuristic
    ranked_list = []
    # adding point closest to mean as the first entry
    ranked_list.extend([closest_point.to_numpy().tolist()])
    # adding the convex hull vertex farthest from the point closest to mean
    point = model.iloc[hull.vertices].iloc[np.argmax(dist_point_vertices)]
    ranked_list.extend([point.to_numpy().tolist()])
    hull_vertex_added.append(np.argmax(dist_point_vertices))

    curr_idx = np.argmax(dist_point_vertices)
    while len(model) > len(ranked_list) and len(hull_vertex_added) < len(hull.vertices) + 1:
        candidates = np.argsort(hull_dist[curr_idx])[1:][::-1]
        for i in range(len(candidates)):
            if candidates[i] not in hull_vertex_added:
                curr_idx = candidates[i]
                break
        point = model.iloc[hull.vertices].iloc[curr_idx]
        ranked_list.extend([point.to_numpy().tolist()])
        hull_vertex_added.append(curr_idx)

    for idx in hull_vertex_added:
        incs_added.append(hull.vertices[idx])

    model = model.drop(index=incs_added).sample(frac=1)
    model = pd.DataFrame(ranked_list, columns=model.columns).append(model, ignore_index=True)
    model = model.drop_duplicates(ignore_index=True)

    return model


model_list = ['RF', 'DT', 'MLP-sgd', 'MLP-adam', 'SVM', 'linear', 'lasso', 'ada', 'kNN', 'linearC']
incumbents = {}
for model_name in model_list:
    if model_name == 'linearC':
        incumbents['LR'] = create_order(model_name)
    else:
        incumbents[model_name] = create_order(model_name)

with open('submissions/switching-optimizer/utils/incumbents_v5.pkl', 'wb') as f:
    pickle.dump(incumbents, f)
