

# Methods
Machine learning modeling was performed using a tree-based gradient 
boosting regression algorithm named CatBoost. The CatBoost algorithm 
is adept at modeling complex processes and has been utilized in previous
adsorption studies. A dataset of 238 data points was compiled in an 
Excel spreadsheet to analyze the removal of Cr(VI) ions from wastewater. 
The dataset includes 23 input variables categorized into four groups: 
Nb2CTx synthesis conditions, elemental composition, physical 
characteristics, and experimental adsorption conditions. Synthesis 
conditions are indicated by parameters such as 'PMS_concentration g/L'.
Physical characteristics are denoted by columns like 'pore_volume_cm3/g' 
and 'avg_pore_width_nm'. The elemental composition would include the 
percentages of elements such as carbon, sulphur, cadmium, maganese, copper,
aluminium, titanium, while experimental 
adsorption conditions are represented by 'time_min' for contact time, 
'Co (initial content of DS pollutant)' for initial Cr(VI) concentration,
'pH', 'catalyst dosage_g/L' for adsorbent dosage, and 'cycle_no' for the 
number of cycles. Target output variables are the removal efficiency, 
represented by the 'removal%' column, and 'K Reaction Rate'. The dataset was 
preprocessed for machine learning by encoding the 'ion_type' and 
'Catalyst' feature. To ensure the robustness and generalizability of 
the model, a 5-fold cross-validation technique was employed, splitting 
the dataset into five equal-sized sub-groups for iterative training and 
validation. Model is trained on four splits and evaluated on the 5th 
split in each iteration. 

# Results
Regression plot shows performance of CatBoost model for the removal
efficiency (%) and K Reaction rate constant (k 10-2min-1) prediction. The distribution of experimental and
CatBoost predicted data is depicted along the marginals using histograms and ridge plots. In
the regression plot, all the scatter points cluster near the solid blackish line, suggesting
a strong correlation between the experimentally calculated and ML-predicted values. The
cross-validation R2 of CatBoost model was 0.98 for  removal efficiency and 0.88 for 
K Reaction rate constant (k 10-2min-1). Additionally, the residual plot 
presents residuals close to the red dashed line have no error. Residuals more
far from the line have more errors. Similarly, the cumulative probability 
plot for removal % and K Reaction rate constant (k 10-2min-1) is a visual representation  
of Absolute error with both outputs. SHAP plots are used for model interpretation, 
elucidating the influence of different input features on the prediction 
of the output variables "Removal Efficiency" and "Rate Constant". This plot 
contributes to the overall model's transparency and interpretability. 
The Shap plot shows that parameter "Time (min)" has most impact in case 
of both output variables.

