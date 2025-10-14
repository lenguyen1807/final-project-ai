#!/bin/bash
set -e
set -x

# -----------------------------
# Params
# -----------------------------
REPORTS_PATH=${1:-"/kaggle/input/reports.csv"}
LABELS_PATH=${2:-"/kaggle/working/labeled_reports.csv"}

# -----------------------------
# Install Python dependencies
# -----------------------------
pip install --upgrade pip
pip install -r requirements.txt

# Install NegBio from GitHub
pip install git+https://github.com/ncbi-nlp/NegBio.git

# -----------------------------
# Download NLTK data
# -----------------------------
python -m nltk.downloader universal_tagset punkt wordnet

# -----------------------------
# Download GENIA+PubMed parser
# -----------------------------
python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"

# -----------------------------
# Clone CheXpert-labeler repo
# -----------------------------
if [ ! -d "chexpert-labeler" ]; then
    git clone https://github.com/stanfordmlgroup/chexpert-labeler.git
fi
cd chexpert-labeler

# -----------------------------
# Run labeling
# -----------------------------
python label.py --reports_path "$REPORTS_PATH" --output_path "$LABELS_PATH"
