conda create -n soft-moe-llava python=3.10 -y
conda activate soft-moe-llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
