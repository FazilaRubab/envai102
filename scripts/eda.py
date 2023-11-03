"""
==========================
Exploratory Data Analysis
==========================
"""

from easy_mpl import plot
from easy_mpl.utils import create_subplots

import matplotlib.pyplot as plt

from utils import LABEL_MAP
from utils import print_version_info
from utils import load_data, get_data

# %%

print_version_info()

# %%

df = load_data()

df.shape

# %%

df.columns

# %%

df.head()

# %%

df.tail()

# %%

df.isna().sum()

# %%

data = get_data()

# %%
print(data.describe())

# %%
data_num = data[[col for col in data.columns if col not in ['ion_type', 'Catalyst']]]
fig, axes = create_subplots(data_num.shape[1], figsize=(10, 8))

for ax, col, label  in zip(axes.flat, data_num, data.columns):

    plot(data_num[col].values, ax=ax, ax_kws=dict(ylabel=LABEL_MAP.get(col, col)),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

