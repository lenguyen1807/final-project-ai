#!/bin/bash
set -e
set -x

# -----------------------------
# Install dependencies
# -----------------------------
sudo apt-get install swig -y
pip install --upgrade pip
pip install -q git+https://github.com/huggingface/transformers.git
pip install bllipparser timm nltk peft==0.10.0 pyyaml bitsandbytes accelerate evaluate salesforce-lavis sentencepiece rouge_score bert-score rad-dino

# Install NegBio from GitHub
git clone https://github.com/ncbi-nlp/NegBio.git
export PYTHONPATH=NegBio:$PYTHONPATH

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