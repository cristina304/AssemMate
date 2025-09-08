#!/bin/bash
# setup_env.sh

pip install \
absl-py==2.3.1 \
accelerate==1.8.1 \
bitsandbytes==0.46.1 \
certifi==2025.6.15 \
charset-normalizer==3.4.2 \
contourpy==1.3.0 \
cycler==0.12.1 \
filelock==3.13.1 \
fonttools==4.58.5 \
fsspec==2024.6.1 \
grpcio==1.73.1 \
huggingface-hub==0.33.2 \
idna==3.10 \
importlib-metadata==8.7.0 \
importlib-resources==6.5.2 \
jinja2==3.1.4 \
kiwisolver==1.4.7 \
markdown==3.8.2 \
markupsafe==2.1.5 \
matplotlib==3.9.4 \
mpmath==1.3.0 \
networkx==3.2.1 \
numpy==1.26.3 \
packaging==25.0 \
pandas==2.3.0 \
peft==0.16.0 \
pillow==11.0.0 \
protobuf==6.31.1 \
psutil==7.0.0 \
pyparsing==3.2.3 \
python-dateutil==2.9.0.post0 \
pytz==2025.2 \
pyyaml==6.0.2 \
regex==2024.11.6 \
requests==2.32.4 \
safetensors==0.5.3 \
sympy==1.13.3 \
tensorboard==2.19.0 \
tensorboard-data-server==0.7.2 \
tokenizers==0.21.2 \
torch==2.7.1+cu118 \
torchaudio==2.7.1+cu118 \
torchvision==0.22.1+cu118 \
tqdm==4.67.1 \
transformers==4.53.1 \
triton==3.3.1 \
typing-extensions==4.12.2 \
urllib3==2.5.0 \
werkzeug==3.1.3

python -c "import torch, transformers, accelerate, bitsandbytes; print('All imports succeed')"
