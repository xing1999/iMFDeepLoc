import numpy as np
import xgboost
import shap
import matplotlib.pyplot as plt
import pandas as pd
shap.initjs()


normalized_data = pd.read_excel(r'E:\Users\zhangzhen\Desktop\Protein and subcellular segmentation\FEC-Net_ALL_data.xlsx', engine='openpyxl')
X = normalized_data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
y = normalized_data.iloc[:, 12]

model = xgboost.RandomForestRegressor().fit(X, y)
# model = xgboost.XGBClassifier(n_estimators=100, max_depth=2).fit(X, y)


# explain the model's predictions using SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
plt.figure(figsize=(10, 6))
shap.waterfall_plot(shap.Explanation(values=shap_values[2020], base_values=explainer.expected_value, data=X.iloc[2020], feature_names=X.columns), max_display=20, show=False)
plt.tight_layout()
plt.show()

# visualize the first prediction's explanation with a force plot
shap.summary_plot(shap_values, X)
plt.show()

# shap.summary_plot(shap_values, X, plot_type="bar")
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.tight_layout()
plt.show()

# visualize the first prediction's explanation with a force plot and save it as HTML file
# plt.rcParams.update({'font.size': 18})
shap.plots.force(shap_values[2020], show=False, matplotlib=True, figsize=(26, 5))
plt.tight_layout()
plt.show()

# create a dependence scatter plot to show the effect of a single feature across the whole dataset
shap.plots.scatter(shap_values[:,"Haralick_836"], color=shap_values[:,"Color_30"])
shap.plots.scatter(shap_values[:,"Haralick_836"], color=shap_values[:,"Gabor_50"])
shap.plots.scatter(shap_values[:,"Color_30"], color=shap_values[:,"Gabor_50"])
shap.plots.scatter(shap_values[:,"Haralick_836"], color=shap_values[:,"Deep_Feature"])
shap.plots.scatter(shap_values[:,"Haralick_836"], color=shap_values[:,"Cavity_30"])
shap.plots.scatter(shap_values[:,"Haralick_836"], color=shap_values[:,"Tario_4"])
shap.plots.scatter(shap_values[:,"Deep_Feature"], color=shap_values[:,"Color_30"])
shap.plots.scatter(shap_values[:,"Cavity_30"], color=shap_values[:,"Color_30"])
shap.plots.scatter(shap_values[:,"Tario_4"], color=shap_values[:,"Color_30"])
shap.plots.scatter(shap_values[:,"Gabor_50"], color=shap_values[:,"LBP_revolve_36"], show=False)
plt.tight_layout()
plt.show()


shap.plots.scatter(shap_values[:, "Haralick_836"])
shap.plots.scatter(shap_values[:, "Color_30"])
shap.plots.scatter(shap_values[:, "Gabor_50"])
shap.plots.scatter(shap_values[:, "Deep_Feature"])
shap.plots.scatter(shap_values[:, "Cavity_30"])
shap.plots.scatter(shap_values[:, "Tario_4"])
shap.plots.scatter(shap_values[:, "LBP_revolve_36"])
shap.plots.scatter(shap_values[:, "LBP_base_256"])
shap.plots.scatter(shap_values[:, "HOG_270"])
shap.plots.scatter(shap_values[:, "HOG_1764"])
plt.tight_layout()
plt.show()

# 热图
explainer = shap.Explainer(model, X)
shap_values = explainer(X[:3727])
shap.plots.heatmap(shap_values, feature_values=shap_values.abs.max(0), max_display=13, show=False)
plt.tight_layout()
plt.show()