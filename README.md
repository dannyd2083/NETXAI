# NETXAI
This project replicates and extends the paper:
"Integrating Explainable AI for Effective Malware Detection in Encrypted Network Traffic"
by evaluating various machine learning models on flow-based network features.

We compare two types of feature sets:

* **CICFlowMeter-generated features**

* **Manually extracted basic features**: a custom set of 9 flow-level features designed to test lightweight alternatives.

## 1. Clone the Repo   
`git clone git@github.com:dannyd2083/NETXAI.git`   
`cd NETXAI `  

## 📦 2. Set up the Environment
Install dependencies via:
`pip install -r requirements.txt`
**We recommend using Python 3.12 and a virtual environment.**



## 📂 3. Dataset Setup

**⚠️ The CTU-13 dataset is very large  and takes a long time to download.**
To avoid waiting, we recommend manually downloading from:   
https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/   
and extracting the dataset into the following path:   
`datasets/ctu13/`
* If not already downloaded, the following script(you don't need to run it) will attempt to download via script (⚠️ slow):
`download_and_setup_data.py`



## 📂 Project Structure & File Roles

| File / Folder	| Description |
| --- | --- |
| `main.py` | Entry point of the full pipeline; accepts CLI arguments to choose feature set and models |
| `requirements.txt` | Lists Python packages required |
| `download_and_setup_data.py` | Downloads `CTU-13` and GitHub feature set; unzips & organizes |
| `process_raw_binetflow.py` | Converts `.binetflow` files to CSV and classifies into normal/malicious |
| `extract_features_basic.py` | Extracts our 9 manually selected basic flow features |
| `extract_features_cic.py` | Organizes `CICFlowMeter` CSVs into usable structure |
| `split_dataset.py` | Splits malicious & normal data into train/val/test (70/10/20) | 
| `train_xgb_models.py` | Trains `XGBoost` model and saves `SHAP` explanations & metrics | 
| `train_rf_models.py` | Trains `Random Forest` model and outputs performance | 
| `train_lgbm_models.py` | Trains `LightGBM` model | 
| `train_knn_models.py` | Trains `K-Nearest Neighbors` model |
| `train_lr_models.py` | Trains `Logistic Regression` model |
| `shap_utils.py`	| `SHAP` plot saving utilities |
| `result_utils.py` | save results to csv | 
| `radar_plot_from_results.py` | Plots radar charts comparing all model results |
| `results/` | Folder where all outputs (metrics, SHAP plots, radar charts) are saved |
| `features/`	| Processed feature CSVs from CICFlowMeter GitHub repo and extracted by ourselves |
| `datasets/`	| Stores original datasets |
| `data_splits/` | Store splited dataset: `train/val/test.csv` | 


## 📁 `results/` Directory Structure
```
results/
├── basic/                      # results of models trained on 9 basic features
│   ├── knn_results.csv
│   ├── lgbm_results.csv
│   ├── lr_results.csv
│   ├── rf_results.csv
│   └── xgb_results.csv

├── cicflowmeter/              # results of models trained on CICFlowMeter features
│   ├── knn_results.csv
│   ├── lgbm_results.csv
│   ├── lr_results.csv
│   ├── rf_results.csv
│   └── xgb_results.csv

├── radar/                     # Radar charts comparing model performance
│   ├── radar_basic.png
│   └── radar_cicflowmeter.png

├── shap/                      # SHAP plots for model explainability
│   ├── basic/                 # SHAP plots for basic features (per model)
│   └── cicflowmeter/         # SHAP plots for CICFlowMeter features (per model)

```











## 🚀 Usage
You can run the full pipeline with:   
`python main.py --feature_set <feature set> --model <model name>`   

Options:

* `--feature_set`: either `basic` or `cicflowmeter` (default: `basic`)

* `--model`: any of `xgb`, `rf`, `lgbm`, `knn`, `lr` (default: all models)   

### 📌 Example Usages

✅ Run the full pipeline with all models on `basic` features:
`python main.py --feature_set basic`

✅ Run the full pipeline with all models on `CICFlowMeter` features:

`python main.py --feature_set cicflowmeter`

✅ Run with selected models only (e.g., `XGBoost`) on basic features:   

`python main.py --feature_set basic --model xgb`

✅ Run with selected models only (e.g., `knn`) on basic features:   

`python main.py --feature_set basic --model knn`

✅ Run with selected models only (e.g., `rf`) on basic features:   

`python main.py --feature_set basic --model rf`

✅ Run with selected models only (e.g., `Light GBM`) on basic features:   

`python main.py --feature_set basic --model lgbm`

✅ Run with selected models only (e.g., `Logistic Regress`) on basic features:   

`python main.py --feature_set basic --model lr`

✅ Run with selected models only (e.g., `XGBoost` and `KNN`) on basic features:

`python main.py --feature_set basic --model xgb knn`

### 🔍 Run a single model independently

You can also run each model's training script manually with a specified feature set:

```
# default with basic feature
python train_xgb_models.py
python train_rf_models.py --feature_set cicflowmeter
python train_lgbm_models.py
python train_knn_models.py --feature_set cicflowmeter
python train_lr_models.py
```
