import matplotlib.pyplot as plt
import numpy as np

# Metrics and model performance
labels = ["Accuracy", "Precision", "Recall", "F1-score"]

xgb  = [0.9628, 0.9632, 0.9700, 0.9666]
lgbm = [0.9627, 0.9632, 0.9698, 0.9665]

data = [lgbm, xgb]
model_labels = ["LightGBM", "XGBoost"]
colors = ["#2ca02c", "#ff7f0e"]

def make_radar_chart(data, labels, model_labels, colors):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, d in enumerate(data):
        values = d + d[:1]
        ax.plot(angles, values, label=model_labels[idx],
                color=colors[idx], linewidth=2.5, marker='o')
        ax.fill(angles, values, color=colors[idx], alpha=0.1)

        # Annotate each point with value
        for i, value in enumerate(values[:-1]):
            ax.text(angles[i], value + 0.003, f"{value:.3f}",
                    color=colors[idx], ha='center', fontsize=8)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.962, 0.97)
    #ax.set_yticks([0.9, 0.92, 0.94, 0.96, 0.98, 1.0])
    #ax.set_yticklabels(['0.90', '0.92', '0.94', '0.96', '0.98', '1.00'])

    ax.set_title("Model Comparison: XGBoost vs LightGBM", size=14, pad=20)
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    # Optional: save to file
    # plt.savefig("radar_xgb_lgbm_knn.png", dpi=300)
    plt.show()

make_radar_chart(data, labels, model_labels, colors)