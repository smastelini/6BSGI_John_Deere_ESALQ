import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feat_eng import MyPipeline


def plot_importances(rf, fig_name):
    importances = rf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in pipeline_A._rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    fig.suptitle('RF feature importance')
    ax.bar(
        range(len(feature_names)), importances[indices], color='gray',
        # yerr=std[indices],
        align='center'
    )
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlim([-1, len(feature_names)])
    plt.xticks(rotation=90)
    plt.ylabel('Importance')
    plt.tight_layout()

    plt.savefig(fig_name)


with open('models/pipeline_A.mdl', 'rb') as f:
    pipeline_A = pickle.load(f)

with open('models/pipeline_B.mdl', 'rb') as f:
    pipeline_B = pickle.load(f)


feature_names = list(pd.read_csv('data/calibration_kmeans10_A_Ca.csv'))[:-1]


plot_importances(pipeline_A._rf, 'RF_feature_importance_group_A.png')
plot_importances(pipeline_B._rf, 'RF_feature_importance_group_B.png')
