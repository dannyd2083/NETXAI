import os
import shap
import matplotlib.pyplot as plt


def generate_and_save_shap_plots(model, X_train, X_test, output_dir, sample_index=0):
    """
    Generate and save SHAP beeswarm and waterfall plots.

    Parameters:
    - model: trained model
    - X_train: training features
    - X_test: testing features
    - output_dir: directory to save plots (e.g., "results/shap/basic/xgb/")
    - sample_index: index of the sample for waterfall plot (default: 0)
    """
    print("Generating SHAP plots to {}...".format(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    # Create SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test.iloc[:100])

    # SHAP Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    beeswarm_path = os.path.join(output_dir, "beeswarm.png")
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()
    print(" Saved:", beeswarm_path)

    # SHAP Waterfall for first sample
    plt.figure()
    shap.plots.waterfall(shap_values[sample_index], show=False)
    waterfall_path = os.path.join(output_dir, f"waterfall_{sample_index}.png")
    plt.savefig(waterfall_path, bbox_inches="tight")
    plt.close()
    print(" Saved:", waterfall_path)
