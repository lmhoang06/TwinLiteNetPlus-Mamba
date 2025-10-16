Install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
bash ~/Miniconda3-latest-Linux-x86_64.sh
```

Create conda environment
```
conda create -n CS311 python==3.10
conda activate CS311
conda install cuda -c nvidia/label/cuda-12.4.1
```
Installing PyTorch (CUDA 12.4)
```
pip install numpy==1.26.3
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
Installing requirements
```
pip install -r requirements.txt
```

Build kernel
```
cd model/kernels/selective_scan
pip install .
```