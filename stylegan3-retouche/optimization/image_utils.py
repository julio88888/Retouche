import torch
import numpy as np
from PIL import Image

def generate_stylegan3_image(G, device, latent_init, truncation_psi=1.0):
    if isinstance(latent_init, np.ndarray):
        latent_init = torch.from_numpy(latent_init)

    latent_init = latent_init.to(device)

    with torch.no_grad():
        generated_tensor = G(latent_init, None, truncation_psi=truncation_psi)

    image_tensor = (generated_tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    image_array = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image_array = (image_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_array)

    return pil_image
