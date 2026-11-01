# EcoSort — AI-Based Waste Identification & Disposal Guidance

An educational Streamlit capstone prototype. It classifies a photo into one of six material classes using MobileNetV2 transfer learning, then looks up general disposal guidance in an editable rules module.

> Disposal advice is general. Check local municipal rules. The image model cannot assess contamination, resin codes, hazardous contents, or local recyclability.

## Features

- Camera and JPG/JPEG/PNG upload input
- 224 × 224 ImageNet preprocessing and MobileNetV2 inference
- Top prediction, confidence bar, and up to two alternatives
- Abstains below 60% confidence with a cautious “Not sure” response
- Separate editable material-to-guidance rules in `disposal.py`
- Session-only scan history, optional labelled real-world test log, confidence-band summary, and CSV download
- Training curves, per-class precision/recall/F1, and confusion-matrix display
- Optional class-folder image count inspection
- Dataset training and evaluation scripts

## Setup

Use Python 3.10 or newer. In a terminal in this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Run the included trained model

The repository includes `model.pt` and `classes.json`, so you can start the app
without downloading the training dataset or training the model again:

```powershell
streamlit run app.py
```

### Retrain the model

Download the TrashNet dataset and arrange it as one folder per class, for example:

The project was trained from the resized dataset in the [original TrashNet repository](https://github.com/garythung/trashnet). This archive contains 2,527 images: cardboard 403, glass 501, metal 410, paper 594, plastic 482, trash 137.

```text
data/garbage_classification/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Folder names define the model's output classes. Training uses an 80/20 random split (seed 42), training-only augmentation, a frozen-head phase followed by up to three fine-tuning epochs, and saves the best checkpoint by validation accuracy.

```powershell
python train.py --data-dir .\data\garbage_classification --epochs 8 --batch-size 32
streamlit run app.py
```

The first training run downloads the pretrained MobileNetV2 weights through torchvision and needs an internet connection. Artifacts are written beside the scripts: `model.pt`, `classes.json`, `training_history.json`, `training_metadata.json`, and validation `evaluation_results.json`. If no model exists, the app opens in guidance/setup mode and explains that inference is unavailable.

## Evaluation

For a labelled evaluation dataset arranged in the same folder-per-class layout:

```powershell
python evaluate.py --data-dir .\data\held_out_test --model .\model.pt --classes .\classes.json
```

This writes `evaluation_results.json`, including accuracy, macro and per-class metrics, support counts, and the confusion matrix. Prefer a truly held-out test set: the training validation set selects the best checkpoint and therefore does not provide an unbiased final estimate. In the app, scan real-world examples, record their known class, then download the session log to document results.

## Windows desktop executable

After training, build the local one-folder Windows bundle:

```powershell
python -m PyInstaller --clean --noconfirm ecosort.spec
```

The executable is `dist/EcoSort/EcoSort.exe`. Keep the full `dist/EcoSort` folder together when moving it to another Windows computer. The program starts the local Streamlit server and opens EcoSort in a browser. Because PyTorch and Streamlit are bundled, the folder is large; it does not require a separate Python installation on that computer.

## Uploading to GitHub

The `.gitignore` keeps local environments, training data, build output, and ZIP
archives out of normal source commits. The trained model and its class list remain
included so a clone can run predictions after installing `requirements.txt`.

1. Create an empty repository on GitHub.
2. Upload the project source files, `.gitignore`, `README.md`, `model.pt`,
   `classes.json`, and the small training/evaluation JSON reports. Do not upload
   `.venv`, `data`, `build`, `dist`, or `EcoSort-Windows-Share.zip` as repository
   files.
3. To share the prebuilt Windows app, open the repository's **Releases** page,
   create a release, and attach `EcoSort-Windows-Share.zip` as a release asset.
   After download, the recipient should extract the ZIP and run
   `EcoSort/EcoSort.exe`, keeping the entire `EcoSort` folder together.

GitHub's browser uploader limits individual repository files to 25 MiB and
regular Git commits block files above 100 MiB. The Windows ZIP is intended as a
release asset; GitHub allows individual release assets below 2 GiB.

## Customization

- Adjust the abstention threshold near the top of `app.py` (`CONFIDENCE_THRESHOLD`). The logbook default is 0.60.
- Edit the material routing and preparation notes in `disposal.py`; verify every rule against the target municipality.
- The default supported classes are the six TrashNet categories. If class names differ, update `disposal.py` mappings or add guidance records; unknown names use the cautious fallback.

## Privacy and limitations

The app does not request an account or location. It processes the submitted image in memory and does not save the photo. Session history is kept in Streamlit session state and is cleared when that session ends; users can export the text log as CSV. Do not present predictions as certain or as a replacement for local waste-management authorities.

## Capstone source inventory

- `app.py` — Streamlit user experience, inference, session records, results/evaluation panels
- `train.py` — dataset loading, preprocessing, transfer learning and checkpoint saving
- `evaluate.py` — evaluation metrics and confusion matrix
- `disposal.py` — separate editable disposal knowledge layer
- `requirements.txt` — dependencies
