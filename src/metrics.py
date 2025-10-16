import csv
import os
import subprocess
import sys

import evaluate
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Define paths for the tools we need to clone and set up
KAGGLE_WORKING_DIR = os.environ.get("KAGGLE_DIR", "/kaggle/working")
METRICS_CACHE_DIR = os.path.join(KAGGLE_WORKING_DIR, "./metrics_cache")
CHEXPERT_DIR = os.path.join(KAGGLE_WORKING_DIR, "chexpert-labeler")
NEGBIO_DIR = os.path.join(KAGGLE_WORKING_DIR, "NegBio")
CHEXPERT_LABELER_SCRIPT = os.path.join(CHEXPERT_DIR, "label.py")

# Set environment variables for Hugging Face cache
os.environ["HF_HOME"] = os.path.join(KAGGLE_WORKING_DIR, "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(KAGGLE_WORKING_DIR, "hf_cache")
os.environ["HF_DATASETS_CACHE"] = os.path.join(KAGGLE_WORKING_DIR, "hf_cache")


def run_chexpert_on_reports(reports_list, input_path, output_path):
    # 1. Write reports to a temporary input CSV file.
    try:
        with open(input_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for report in reports_list:
                # Ensure the report is a single string element in the row
                writer.writerow([report])
    except IOError as e:
        print(f"❌ ERROR: Could not write to temporary file {input_path}. Error: {e}")
        return None

    # 2. Construct and run the command-line script as a subprocess.
    if not os.path.exists(CHEXPERT_LABELER_SCRIPT):
        print(
            f"❌ ERROR: CheXpert labeler script not found at '{CHEXPERT_LABELER_SCRIPT}'."
        )
        print(
            "Please run the setup function: `from medblip.metrics import setup_metrics_env; setup_metrics_env()`"
        )
        return None

    command = [
        sys.executable,
        CHEXPERT_LABELER_SCRIPT,
        "--reports_path",
        input_path,
        "--output_path",
        output_path,
    ]
    print(f"INFO: Running command: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )

    if result.returncode != 0:
        print("❌ ERROR: CheXpert labeler script failed.")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return None

    # 3. Read the labeled results back into a pandas DataFrame.
    try:
        labeled_df = pd.read_csv(output_path)
        return labeled_df
    except FileNotFoundError:
        print(f"❌ ERROR: CheXpert output file not found at {output_path}.")
        return None


def compute_clinical_factuality(predictions, references):
    print("INFO: Starting CheXpert-based factuality evaluation...")

    # Define temporary file paths in the writable directory
    tmp_dir = os.path.join(METRICS_CACHE_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    pred_input_path = os.path.join(KAGGLE_WORKING_DIR, "tmp_pred_reports.csv")
    pred_output_path = os.path.join(KAGGLE_WORKING_DIR, "tmp_labeled_preds.csv")
    ref_input_path = os.path.join(KAGGLE_WORKING_DIR, "tmp_ref_reports.csv")
    ref_output_path = os.path.join(KAGGLE_WORKING_DIR, "tmp_labeled_refs.csv")

    try:
        # Run labeling on both prediction and reference reports
        pred_labels = run_chexpert_on_reports(
            predictions, pred_input_path, pred_output_path
        )
        ref_labels = run_chexpert_on_reports(
            references, ref_input_path, ref_output_path
        )

        if pred_labels is None or ref_labels is None:
            return {}

        chexpert_classes = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

        # Ensure all required columns exist, fill with 0 if not
        for col in chexpert_classes:
            if col not in pred_labels.columns:
                pred_labels[col] = 0
            if col not in ref_labels.columns:
                ref_labels[col] = 0

        # Align columns and handle uncertain labels (-1.0 -> 1.0, positive) and fill NaNs
        pred_labels = pred_labels[chexpert_classes].replace(-1.0, 1.0).fillna(0)
        ref_labels = ref_labels[chexpert_classes].replace(-1.0, 1.0).fillna(0)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref_labels, pred_labels, average="micro", zero_division=0
        )

        print("\n--- CheXpert Classification Report (Per-Finding) ---")
        report = classification_report(
            ref_labels,
            pred_labels,
            target_names=chexpert_classes,
            zero_division=0,
            digits=3,
        )
        print(report)
        print("------------------------------------------------------\n")

        return {
            "CheXpert-F1 (micro)": f1,
            "CheXpert-Precision (micro)": precision,
            "CheXpert-Recall (micro)": recall,
        }
    finally:
        # Clean up temporary files
        for path in [
            pred_input_path,
            pred_output_path,
            ref_input_path,
            ref_output_path,
        ]:
            if os.path.exists(path):
                os.remove(path)
        print("INFO: Cleaned up temporary files.")


# The functions for linguistic fluency and the main wrapper remain the same as before.
def compute_linguistic_fluency(predictions, references):
    """Computes standard NLP metrics (BLEU, ROUGE, etc.). (No changes here)"""
    print("INFO: Starting linguistic fluency evaluation...")
    # Add a check for empty lists to avoid errors in evaluate library
    if not predictions or not references:
        print(
            "WARNING: predictions or references list is empty. Skipping linguistic metrics."
        )
        return {}

    formatted_references = [[r] for r in references]

    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    # Compute metrics
    bleu_results = bleu.compute(
        predictions=predictions, references=formatted_references
    )
    rouge_results = rouge.compute(predictions=predictions, references=references)
    meteor_results = meteor.compute(predictions=predictions, references=references)

    # BERTScore can be slow; ensure GPU is used if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"INFO: Running BERTScore on device: {device}")
    bertscore_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="distilbert-base-uncased",
        device=device,
    )

    return {
        "BLEU-1": bleu_results["precisions"][0],
        "BLEU-4": bleu_results["bleu"],
        "ROUGE-L": rouge_results["rougeL"],
        "METEOR": meteor_results["meteor"],
        "BERTScore-F1": np.mean(bertscore_results["f1"]),
    }


def compute_all_metrics(predictions, references, compute_clinical: bool = False):
    # Filter out empty strings which can cause errors in downstream metrics
    filtered_pairs = [
        (p, r)
        for p, r in zip(predictions, references)
        if p and p.strip() and r and r.strip()
    ]
    if not filtered_pairs:
        print(
            "ERROR: No valid prediction/reference pairs with non-empty strings to evaluate."
        )
        return {}

    filtered_predictions, filtered_references = zip(*filtered_pairs)
    filtered_predictions = list(filtered_predictions)
    filtered_references = list(filtered_references)

    print(f"INFO: Computing all metrics for {len(filtered_predictions)} valid samples.")

    fluency_metrics = compute_linguistic_fluency(
        filtered_predictions, filtered_references
    )
    if compute_clinical:
        factuality_metrics = compute_clinical_factuality(
            filtered_predictions, filtered_references
        )
        return {**fluency_metrics, **factuality_metrics}
    return {**fluency_metrics}
