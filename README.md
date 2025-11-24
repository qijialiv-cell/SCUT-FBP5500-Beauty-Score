# SCUT-FBP5500 Beauty Score

This repository contains a fastai notebook that trains a ResNet34-based regression model to predict facial beauty ratings on the SCUT-FBP5500_v2 dataset.

## Repository structure

- `fast_ai_beauty_score.ipynb` — Notebook with data preparation, model training, export, inference and a transfer-learning template.
- `SCUT-FBP5500_v2/` — Dataset folder (not included). Contains `Images/`, `train_test_files/` and facial landmarks.

## Project tree

A concise view of the repository layout:

```
beauty_score/
├─ README.md
├─ fast_ai_beauty_score.ipynb
└─ SCUT-FBP5500_v2/
   ├─ README.txt
   ├─ Images/                # 5500 face images (jpg)
   ├─ train_test_files/      # train/test split files
   └─ facial landmark/       # .pts landmark files
```

## Dataset & Citation

The experiments and data used in this project are based on the SCUT-FBP5500 dataset. If you use this dataset or results from this repository, please cite the original paper:

@article{liang2017SCUT,
title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
jurnal    = {ICPR}, 
year      = {2018}
}

The dataset folder `SCUT-FBP5500_v2/README.txt` also contains the original README and citation notice from the dataset authors.

## Requirements

- Python 3.8+ (or 3.9/3.10)
- fastai (v2/v3 compatible with notebook; install via `pip install fastai`)
- torch (matching fastai requirements; GPU recommended)
- pandas

You can install minimal requirements with:

```bash
pip install fastai pandas
```

If you plan to use the GPU, install a matching `torch` version from https://pytorch.org.

## Quick start (notebook)

1. Open `fast_ai_beauty_score.ipynb` in Jupyter or VS Code Notebook.
2. Set the `data_path` variable near the top of the notebook to point to your local copy of `SCUT-FBP5500_v2` (for example `Path("~/Downloads/SCUT-FBP5500_v2")`).
3. Run cells in order. Key cells:
   - Data preparation: reads `train.txt` / `test.txt` and creates the `DataBlock` and `DataLoaders`.
   - Learner creation: builds a `vision_learner` using `resnet34` for regression (MSE / RMSE metric).
   - Training: `learn.fine_tune(...)` with SaveModel and EarlyStopping callbacks.
   - Export: `learn.export("beauty_model.pkl")` for inference.
   - Inference: load the exported learner and run batch predictions (fast on GPU).

## Transfer learning

The notebook includes a transfer-learning template. To adapt it:

1. Prepare your new dataset with the same image preprocessing (Resize(224) and fastai defaults).
2. Update the `new_data_path`, `new_train_txt`, and `new_valid_txt` variables in the transfer cell.
3. Run the transfer cell: it will load `beauty_model.pkl`, create a new learner for your dataset, copy weights and fine-tune.

Notes:
- If the new dataset has a different head/label format, adjust the model head or load only matching layers.
- Use `cpu=True` when calling `load_learner` if you don't have a GPU.

## Tips

- Verify paths before running (e.g., `data_path.exists()` and `df.head()`).
- Reduce batch size if you run out of GPU memory.
- Save intermediate checkpoints if training is long.

## License

This project is provided as-is for research and learning purposes.
# SCUT-FBP5500-Beauty-Score
