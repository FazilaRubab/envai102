"""
====================
Experiments
====================
"""
import matplotlib.pyplot as plt
from ai4water.experiments import MLRegressionExperiments

from utils import print_version_info
from utils import make_data, set_rcParams, SAVE

# %%

print_version_info()

# %%

set_rcParams()

# %%
# Removal Efficiency
# ======================

data, enc = make_data()

print(data.shape)

output_features = data.columns.tolist()[-1:]
# %%
experiments = MLRegressionExperiments(
    input_features = data.columns.tolist()[0:-1],
    output_features = output_features,
    train_fraction=1.0,
    cross_validator={"KFold": {"n_splits": 5}},
    show=False
)

# %%

experiments.fitcv(
    data=data,
    exclude=[
        "SGDRegressor", 'Lars',
        'LarsCV', 'RANSACRegressor',
        'OneClassSVM',
        'GaussianProcessRegressor',
        'DummyRegressor',
        'LassoLarsCV'
    ],
)

# %%
ax = experiments.plot_cv_scores(fill_color="#2596be", patch_artist=True,
                                exclude="LinearSVR", figsize=(9, 6))
ax.grid()
if SAVE:
    plt.savefig("results/figures/exp_eff.png", bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()

# %%
# Rate Constant
# ==================

data, enc = make_data(
    output_features='K Reaction rate constant (k 10-2min-1)'
)

print(data.shape)

output_features = data.columns.tolist()[-1:]
# %%
experiments = MLRegressionExperiments(
    input_features = data.columns.tolist()[0:-1],
    output_features = output_features,
    train_fraction=1.0,
    cross_validator={"KFold": {"n_splits": 5}},
    show=False
)

# %%

experiments.fitcv(
    data=data,
    exclude=[
        #"SGDRegressor",
        #'Lars',
        #'LarsCV',
        'RANSACRegressor',
        'OneClassSVM',
        #'GaussianProcessRegressor',
        'DummyRegressor',
        #'LinearSVR',
        #'KNeighborsRegressor',
    ]
)

# %%

ax = experiments.plot_cv_scores(
    exclude=["Lars", "LinearSVR", "SGDRegressor"],
    fill_color="#2596be", patch_artist=True,
)
ax.grid()
if SAVE:
    plt.savefig("results/figures/rate_const.png", bbox_inches="tight", dpi=600)
plt.tight_layout()
plt.show()
