import matplotlib.pyplot as plt
import numpy as np

# Metrics and model performance
labels = ["Accuracy", "Precision", "Recall", "F1-score", "MCC"]

xgb    = [0.9628, 0.9632, 0.9700, 0.9666, 0.9246]
lgbm   = [0.9627, 0.9632, 0.9698, 0.9665, 0.9244]
rf     = [0.9196, 0.9357, 0.9184, 0.9269, 0.8379]
knn    = [0.951494, 0.956267, 0.956353, 0.95631, 0.901794]
logreg = [0.74885, 0.778953, 0.764493, 0.771655, 0.492801]

data = [xgb, lgbm, rf, knn, logreg]
model_labels = ["XGB", "LGBM", "RF", "KNN", "LogReg"]
colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"]


# Radar helper function
def make_radar_chart(data, labels, model_labels, colors):
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    # Setup the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each model
    for idx, d in enumerate(data):
        values = d + d[:1]  # close the shape
        ax.plot(angles, values, label=model_labels[idx], color=colors[idx])
        ax.fill(angles, values, color=colors[idx], alpha=0.2)

    # Add labels and customize
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.49, 1.0)  # adjust to zoom into your high scores

    # Draw y-labels
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.show()

# Call the radar chart function
make_radar_chart(data, labels, model_labels, colors)
