import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def plot_maps(data, model, grupo, map_id, prop='Ca'):
    # X, Y = np.meshgrid(data['POINT_X'], data['POINT_Y'])
    X, Y = data['POINT_X'], data['POINT_Y']

    fig, ax = plt.subplots(figsize=(5, 10), nrows=3, dpi=300)

    model_names = {}
    if model == 'rf':
        model_names = {
            'rf': 'RF',
            'gbt_05': 'GBT$_{P0.05}$',
            'gbt_95': 'GBT$_{P0.95}$',
        }
    else:
        model_names = {
            'gpr': 'GPR',
            'gpr_05': 'GPR$_{0.05}$',
            'gpr_95': 'GPR$_{0.95}$',
        }

    Z = [data[f'preds_{name}'] for name in model_names]

    min_ = min([min(Z[0]), min(Z[1]), min(Z[2])])
    max_ = max([max(Z[0]), max(Z[1]), max(Z[2])])

    for i, (name, pname) in enumerate(model_names.items()):

        ax[i].scatter(
            X, Y, c=Z[i], cmap='Oranges',
            vmin=min_, vmax=max_
        )
        ax[i].set_title(pname)
        ax[i].locator_params(axis='x', nbins=5)

    norm = matplotlib.colors.Normalize(vmin=min_, vmax=max_)
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.09, 0.06, 0.84, 0.02])
    fig.colorbar(
        sm,
        orientation="horizontal", cax=cbar_ax
    )

    plt.savefig(f'mapa_{map_id}_{grupo}_{prop}_{model}.png', bbox_inches='tight')


# mapa = pd.read_csv('predictions/malha_A_1024x1024.csv')
# plot_maps(mapa, 'rf', 'A', '1024x1024')
# plot_maps(mapa, 'gpr', 'A', '1024x1024')
#
# mapa = pd.read_csv('predictions/malha_B_1024x1024.csv')
# plot_maps(mapa, 'rf', 'B', '1024x1024')
# plot_maps(mapa, 'gpr', 'B', '1024x1024')

mapa = pd.read_csv('predictions/malha_A_200x200.csv')
plot_maps(mapa, 'rf', 'A', '200x200')
plot_maps(mapa, 'gpr', 'A', '200x200')

mapa = pd.read_csv('predictions/malha_B_200x200.csv')
plot_maps(mapa, 'rf', 'B', '200x200')
plot_maps(mapa, 'gpr', 'B', '200x200')
