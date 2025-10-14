#!/bin/bash
set -e
set -x

# -----------------------------
# Install dependencies
# -----------------------------
sudo apt-get install swig -y
pip install --upgrade pip
pip install bllipparser nltk peft pyyaml bitsandbytes accelerate evaluate salesforce-lavis sentencepiece

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