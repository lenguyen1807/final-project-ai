# 🩺 End-to-End Medical Image Captioning Pipeline

This repository provides a **complete end-to-end pipeline** for training, evaluating, and visualizing **medical image captioning models** — particularly on the **Indiana University Chest X-Ray (IU X-Ray)** dataset.

It supports modular configuration, model setup, PEFT extensions, and flexible decoder selection (`GPT-2`, `T5`, `LLaMA`).

---

## 📦 1. Project Structure

```
medical-image-captioning/
│
├── src/
│   ├── simple_config.py          # Model configuration utilities
│   ├── simple_trainer.py         # Trainer class (data loaders, train/val loops)
│   └── ...                       # Supporting modules (datasets, utils, etc.)
│
├── chest-xrays-indiana-university/
│   ├── train_df.csv
│   ├── val_df.csv
│   └── test_df.csv
│
├── results/                      # Saved outputs (after training)
│
├── end_to_end_pipeline.py        # Main end-to-end pipeline
└── README.md                     # You are here
```

---

## ⚙️ 2. Installation

Make sure your environment has the following dependencies:

```bash
conda create -n medcap python=3.10 -y
conda activate medcap

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # (adjust CUDA version if needed)
pip install pandas numpy matplotlib seaborn pillow tqdm transformers
```

Optionally, for parameter-efficient fine-tuning (PEFT):

```bash
pip install peft bitsandbytes accelerate
```

---

## 📁 3. Dataset Setup

Place the **IU X-Ray dataset** (or similar structured data) under:

```
chest-xrays-indiana-university/
```

Expected CSV format:

| image_path | caption |
|-------------|----------|
| path/to/image1.png | The lungs are clear without evidence of active disease. |
| path/to/image2.png | Mild cardiomegaly noted. |

✅ Ensure all CSV files exist:
- `train_df.csv`
- `val_df.csv`
- `test_df.csv`

Each file must contain at least the columns: `image_path`, `caption`.

---

## 🚀 4. Quick Start — Run Full Pipeline

### Default (GPT-2 decoder)
```bash
python end_to_end_pipeline.py --decoder gpt2 --epochs 3 --batch_size 4
```

### Using T5 or LLaMA as decoder
```bash
python end_to_end_pipeline.py --decoder t5 --epochs 5 --batch_size 8
# or
python end_to_end_pipeline.py --decoder llama --epochs 5
```

### Specify custom dataset directory
```bash
python end_to_end_pipeline.py --data_dir /path/to/IU_XRay
```

---

## 🧠 5. Pipeline Stages

| Stage | Description |
|--------|--------------|
| **Load Data** | Load and validate CSVs; check for missing values & image existence. |
| **Setup Model** | Initialize config and trainer for chosen decoder (ViT-Encoder + GPT2/T5/LLaMA-Decoder). |
| **Train** | Train model with training and validation sets. |
| **Evaluate** | Compute test loss, generate sample captions. |
| **Inference** | Run caption generation on random samples for qualitative results. |
| **Visualize** | Generate plots and caption statistics (`results/training_results.png`). |
| **Save Results** | Export JSON and text summaries to `results/` folder. |

---

## 📊 6. Output Files

After successful execution, the following are generated in the `results/` directory:

| File | Description |
|-------|-------------|
| `pipeline_results.json` | Config + results summary |
| `sample_captions.txt` | Generated captions during evaluation |
| `inference_results.txt` | Ground-truth vs. generated captions |
| `training_results.png` | Visualization of training curves and caption stats |

---

## 🧩 7. Example Visualization

After training, a visualization like this is saved:
```
results/training_results.png
```
It includes:
- Loss curves  
- Caption length distribution  
- Example generated vs. reference captions  

---

## 🧪 8. Reproducibility Notes

- Set random seeds inside your training loop for reproducibility.
- Ensure consistent preprocessing for both training and inference.
- Fine-tune hyperparameters (learning rate, max_length, etc.) inside `simple_config.py`.

---

## 🧰 9. Extending the Pipeline

To extend this codebase:
- Replace encoder/decoder definitions inside `simple_trainer.py`.
- Integrate **LoRA / Prefix Tuning / Prompt Tuning** via the PEFT library.
- Add **BLEU, CIDEr, METEOR, CheXbert** metrics for medical caption evaluation.

---

## 🩻 10. Citation

If you use this pipeline or its adapted components, please cite:
```
@dataset{iu_xray,
  title={Indiana University Chest X-ray Dataset},
  author={Demner-Fushman et al.},
  year={2015},
  url={https://openi.nlm.nih.gov/}
}
```

---

## ✅ Example Command Recap

| Model | Example Command |
|--------|------------------|
| GPT-2 | `python end_to_end_pipeline.py --decoder gpt2` |
| T5 | `python end_to_end_pipeline.py --decoder t5 --epochs 5 --batch_size 8` |
| LLaMA | `python end_to_end_pipeline.py --decoder llama --epochs 3` |

---

**🎉 Done!**  
After training, open the `results/` folder for generated captions, loss curves, and metrics.
