"""
====================
main
====================
"""

from utils import make_data, set_rcParams, regression_plot, get_predictions
from utils import residual_plot, shap_plot, print_version_info
from utils import cumulative_probability_plot

# %%
print_version_info()

# %%
set_rcParams()

# %%
data, enc = make_data()
print(data.shape)

# %%
# Removal Efficiency
# ======================

# %%
# This function is called to obtain 3 variables. The use of "KFold" with "n_splits=5" suggests
# that the data might be divided into five subsets for cross-validation during model training
# and evaluation to ensure robustness and reliability of the predictions.
rem_true, rem_pred, rem_scores = get_predictions("removal%", cv="KFold", n_splits=5)
rem_pred[rem_pred<0.0] = 0.0 # todo

# %%
# Scatter plot illustrating the strong linear relationship between predicted and
# experimental 'removal%' values (R²=0.98) with a low RMSE of 4.25.

regression_plot(rem_true, rem_pred, hist_bins=25,  label="Removal %")

# %%
# This residual plot for removal percentage serves as a critical visualization in manufacturing
# and quality control. It presents the discrepancies, or residuals, in removal % of
# defective items within a production cycle. This graphical representation aids in the detection
# of irregularities or trends in these residuals, shedding light on variations or inconsistencies
# within the manufacturing process. Through a thorough analysis of this plot, manufacturers can
# precisely pinpoint the timing and locations where defects occur, facilitating targeted
# improvements and quality control measures to curtail removal % and enhance
# overall product quality. It functions as an indispensable tool for streamlining production
# processes and minimizing waste.

residual_plot(rem_true, rem_pred, x_axis_label= "Removal %")

# %%
# This error plot for removal % is a visual representation used in quality control
# and manufacturing analysis. It displays the discrepancies or errors in removal %
# calculations for defective items during production. This plot is essential for
# identifying deviations from expected values, enabling manufacturers to pinpoint
# areas in the process where quality issues occur and take corrective actions to
# reduce errors and enhance product quality.

cumulative_probability_plot(rem_true, rem_pred, x_axis_label= "Removal %")

# %%
# This SHAP plot for "removal%" leverages the provided label mapping to display
# the influence of different features on the removal percentage in a comprehensible manner.
# It visually represents the Shapley values, quantifying the contribution of each feature
# to the predicted outcome. By examining this plot, you can easily identify which factors,
# such as "PMS Concentration (g/L)" or "Catalyst Dosage (g/L)," have the most significant
# impact on the removal percentage, aiding in process optimization and quality control efforts.
shap_plot(output_features= "removal%")

# %%
# Rate Constant
# ==================
k_true, k_pred, k_scores = get_predictions("K Reaction rate constant (k 10-2min-1)",
                                           cv="KFold",
                                              n_splits=5)
k_pred[k_pred<0.0] = 0.0 # todo

# %%
# Regression plot depicting the relationship between predicted and experimental
# Rate Constant (k) values, showing a moderate fit with R-squared (R²) of 0.88
# and a RMSE of 0.56.

regression_plot(k_true, k_pred, hist_bins=25,  label="Rate Constant (k)")


# %%
# Residual plot
residual_plot(k_true, k_pred, x_axis_label= "Rate Constant (k)")

# %%
# Error Plot
cumulative_probability_plot(k_true, k_pred, x_axis_label= "Rate Constant (k)")

# %%
# Shap Plot
shap_plot(output_features= "K Reaction rate constant (k 10-2min-1)")






