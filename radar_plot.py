import matplotlib.pyplot as plt
import numpy as np
import os

def make_radar_chart(data, labels, model_labels, title=None, save_path=None, colors=None):
    """
    Plot a radar chart to compare models based on performance metrics.

    Parameters:
    - data: list of lists. Each list contains metric values for a model.
    - labels: list of metric names.
    - model_labels: list of model names (same length as data).
    - title: optional title string.
    - save_path: if given, save the plot as an image file.
    - colors: optional list of colors for lines.
    """
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    if not colors:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, model_data in enumerate(data):
        values = model_data + model_data[:1]
        ax.plot(angles, values, label=model_labels[i], color=colors[i % len(colors)], linewidth=2)
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_rlabel_position(0)

    if title:
        plt.title(title, y=1.08, fontsize=14)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print("Radar plot saved to: {}".format(save_path))
    else:
        plt.show()