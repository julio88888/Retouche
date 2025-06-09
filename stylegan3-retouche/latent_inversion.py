# latent_inversion.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import vgg19
from PIL import Image
import numpy as np
import cv2
import scipy.ndimage
from face_alignment import FaceAlignment


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features[:29].eval().to(device)
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.transforms = transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])

    def forward(self, pred, target, weight_mask):
        pred = self.transforms((pred + 1) / 2)
        target = self.transforms((target + 1) / 2)
        pred_masked = pred * weight_mask
        target_masked = target * weight_mask
        pred_feats = self.vgg(pred_masked)
        target_feats = self.vgg(target_masked)
        return nn.functional.mse_loss(pred_feats, target_feats)


def preprocess_image(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img).unsqueeze(0)


def generate_face_weight_mask(image_tensor, image_size):
    fa = FaceAlignment('2D', device='cuda' if torch.cuda.is_available() else 'cpu')
    img_np = ((image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    preds = fa.get_landmarks(img_np)
    if preds is None:
        raise ValueError("Aucun visage détecté.")
    landmarks = preds[0]
    hull = cv2.convexHull(landmarks.astype(np.int32))
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    blurred = scipy.ndimage.gaussian_filter(mask.astype(np.float32), sigma=25)
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min())
    weighted = 0.2 + 0.6 * blurred
    return torch.tensor(weighted, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)


def save_generated_image(tensor, filename):
    img = (tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)


def image_to_latent(target_image, generator, num_steps=1600, learning_rate=0.10,
                    latent_dim=512, checkpoint_dir='checkpoints', checkpoint_interval=200):
    os.makedirs(checkpoint_dir, exist_ok=True)
    z = torch.randn(1, latent_dim, requires_grad=True, device=device)
    optimizer = optim.Adam([z], lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    weight_mask = generate_face_weight_mask(target_image, target_image.shape[-1])
    weight_mask_rgb = weight_mask.expand_as(target_image)

    mse_loss = nn.MSELoss()
    perceptual_loss = VGGPerceptualLoss()

    for step in range(num_steps):
        optimizer.zero_grad()
        generated_img = generator(z, None, truncation_psi=1)

        loss_mse = ((generated_img - target_image) ** 2 * weight_mask_rgb).mean()
        loss_vgg = perceptual_loss(generated_img, target_image, weight_mask_rgb)
        loss = loss_mse + 0.3 * loss_vgg

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(f"Step {step} | MSE: {loss_mse.item():.4f} | VGG: {loss_vgg.item():.4f}")

        if step % checkpoint_interval == 0:
            save_generated_image(generated_img, os.path.join(checkpoint_dir, f"step_{step}.png"))

    return z.detach()
