# PlantVillage Plant Disease Classification

Image classification models (CNN baselines + Vision Transformer via `timm`) trained on the PlantVillage leaf disease dataset, with duplicate/near-duplicate image detection, stratified train/val/test splitting, and model comparison (accuracy, macro F1, confusion matrices).

## Requirements

- Python 3.9+
- A CUDA-capable GPU is recommended (falls back to CPU automatically)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This notebook uses the **PlantVillage** dataset (Kaggle: [emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)).

**Option A — Kaggle API (recommended)**
1. Get your `kaggle.json` API token from your Kaggle account settings.
2. Place it at `~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows) and restrict permissions:
   ```bash
   mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. Download and unzip:
   ```bash
   kaggle datasets download -d emmarex/plantdisease -p ./plantvillage --unzip
   ```
4. Update `DATA_ROOT` in the notebook to point to the extracted folder (e.g. `./plantvillage/PlantVillage`).

**Option B — Manual download**
Download the dataset directly from Kaggle, unzip it locally, and set `DATA_ROOT` in the notebook to that folder.

> Note: the notebook was originally written for Google Colab and includes a `google.colab.drive.mount(...)` cell. If running locally or on another platform, skip/remove that cell and just set `DATA_ROOT` to your local dataset path.

## Running

1. Clone the repo:
   ```bash
   git clone https://github.com/TshifTAJ/PlantVillage-Plant-Disease-Classification.git
   cd PlantVillage-Plant-Disease-Classification
   ```
2. Install dependencies (see above).
3. Set up the dataset (see above) and update `DATA_ROOT` in the first cells of `PlantVillage_Classification.ipynb`.
4. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook PlantVillage_Classification.ipynb
   ```

### Or run in Google Colab
Open the notebook in Colab, mount your Google Drive, upload the PlantVillage folder to Drive (or use the Kaggle API cell), and run all cells.

## Outputs

Running the notebook produces:
- `model_comparison_summary.csv` — accuracy / macro F1 per model
- `confusion_matrix_<model>.png` — one confusion matrix per trained model

#Requirements

torch
torchvision
timm
imagehash
numpy
pandas
scikit-learn
matplotlib
seaborn
Pillow
kaggle
