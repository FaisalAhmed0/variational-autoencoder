# Auto-Encoding Variational Bayes
This repo. contains a reimplementation of the variational auto encoder based on the original paper "Auto-Encoding Variational Bayes". by Kingma et.al<br/>
Paper Link: https://arxiv.org/abs/1312.6114

### Clone the repository

```bash
git clone https://github.com/FaisalAhmed0/variational-autoencoder
```

### you can setup a new environment and install requirements.txt

```bash
conda create -n vae_env 
pip3 install -r requirements.txt 
```

### activate the new environment and run train.py

```bash
conda activate vae_env
python train.py --dataset mnist
```
