
import os
import numpy as np
import torch
import PIL.Image
import pandas as pd
import dnnlib
import legacy

def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def load_generator(url, device):
    with dnnlib.util.open_url(url) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

def seed2vec(G, seed):
    return np.random.RandomState(seed).randn(1, G.z_dim)

def get_label(G, device, class_idx=None):
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0 and class_idx is not None:
        label[:, class_idx] = 1
    return label

def generate_image(device, G, z, truncation_psi=0.6, noise_mode='const', class_idx=None):
    z = torch.from_numpy(z).to(device)
    label = get_label(G, device, class_idx)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    img = img.resize((244, 244), PIL.Image.LANCZOS)
    return img
