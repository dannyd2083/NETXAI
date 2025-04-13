import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_results_from_folder(folder_path):
    """
    Load all results from a given results folder
    Return model names and corresponding score matrix
    """
    result_files = sorted([f for f in os.listdir(folder_path) if f.endswith("_results.csv")])
    data = []
    model_names = []

    for file in result_files:
        df = pd.read_csv(os.path.join(folder_path, file), index_col=0)
        model_name = file.replace("_results.csv", "").upper()
        model_names.append(model_name)
        scores = df.iloc[:, 0].values  # assumes 1 column
        data.append(scores)

    return model_names, df.index.tolist(), np.array(data)

def make_radar_chart_basic(data, labels, model_labels, feature_set, save_path):
    """
    Create radar chart from matrix and save the figure
    """
    num_vars = len(labels)
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"]

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    # Setup the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # # Plot each model
    # print("data:\n", data)
    # print("model labels\n", model_labels)
    for idx, d in enumerate(data):
        values = d.tolist() + d[:1].tolist()  # close the shape
        color_i = colors[idx % len(colors)]
        ax.plot(angles, values, label=model_labels[idx], color=color_i)
        ax.fill(angles, values, color=color_i, alpha=0.2)

    title=f"Model Performance on {feature_set.capitalize()} Features"
    # Add labels and customize
    ax.set_title(title, size=12)
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
    plt.savefig(save_path, bbox_inches='tight')
    print("Radar chart saved to:", save_path)
    plt.show()
    plt.close()
    

def make_radar_chart_cic(data, labels, model_labels, feature_set, save_path):
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
    num_vars = len(labels)

    # Compute angle of each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the circle

    # Setup the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Plot each model
    for idx, d in enumerate(data):
        values = d.tolist() + d[:1].tolist()  # close the shape
        color_i = colors[idx % len(colors)]
        ax.plot(angles, values, label=model_labels[idx], color=color_i)
        ax.fill(angles, values, color=color_i, alpha=0.2)

    title=f"Model Performance on {feature_set.capitalize()} Features"
    # Add labels and customize
    ax.set_title(title, size=12)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.84, 1.0)  # adjust to zoom into your high scores

    # Draw y-labels
    ax.set_rlabel_position(0)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print("Radar chart saved to:", save_path)
    plt.show()
    plt.close()
    



def main():
    output_dir = os.path.join("results", "radar")
    os.makedirs(output_dir, exist_ok=True)

    for feature_set in ["basic", "cicflowmeter"]:
        folder = os.path.join("results", feature_set)
        model_names, metric_labels, score_matrix = load_results_from_folder(folder)
        save_path = os.path.join(output_dir, f"radar_{feature_set}.png")
        if feature_set == "basic":
            make_radar_chart_basic(score_matrix, metric_labels, model_names, feature_set, save_path)
        else:
            make_radar_chart_cic(score_matrix, metric_labels, model_names, feature_set, save_path)


if __name__ == "__main__":
    main()
