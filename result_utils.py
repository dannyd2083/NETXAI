import os
import pandas as pd
from radar_plot import make_radar_chart


def save_results_to_csv(results_dict, metric_names, save_path):
    """
    Save evaluation results to CSV.
    
    Parameters:
    - results_dict: dict, {model_name: [acc, prec, rec, f1, mcc], ...}
    - metric_names: list of metric names
    - save_path: path to save the CSV
    """
    df = pd.DataFrame(results_dict, index=metric_names)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    print("Saved results to: {}".format(save_path))


def generate_radar_plot_from_csv(csv_path, title=None, save_path=None):
    """
    Load results from CSV and generate radar chart.
    
    Parameters:
    - csv_path: path to the results CSV
    - title: radar chart title
    - save_path: image path to save the plot
    """
    if not os.path.exists(csv_path):
        print("[Error]: File not found - {}".format(csv_path))
        return

    df = pd.read_csv(csv_path, index_col=0)
    metric_labels = df.index.tolist()
    model_labels = df.columns.tolist()
    data = [df[model].tolist() for model in model_labels]

    make_radar_chart(
        data=data,
        labels=metric_labels,
        model_labels=model_labels,
        title=title,
        save_path=save_path
    )
