import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
print("FILES IN FOLDER:", os.listdir())

from models.resnet import load_model
from src.preprocess import preprocess_image
from src.fgsm import fgsm_attack

print("Initializing PRISM-Med (PyTorch Version)...")

model = load_model()

def run_experiment(image_path, epsilon=0.02):
    # Load image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image)

    # Original prediction
    output = model(input_tensor)
    orig_idx = output.argmax(dim=1).item()

    # Attack
    adv_img, noise = fgsm_attack(model, input_tensor, epsilon)

    # Adversarial prediction
    adv_output = model(adv_img)
    adv_idx = adv_output.argmax(dim=1).item()

    # Convert tensors to displayable format
    orig_img_np = input_tensor.squeeze().permute(1, 2, 0).detach().numpy()
    adv_img_np = adv_img.squeeze().permute(1, 2, 0).detach().numpy()
    noise_np = noise.squeeze().permute(1, 2, 0).detach().numpy()

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Original\nClass: {orig_idx}")
    plt.imshow(orig_img_np)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Adversarial Noise")
    plt.imshow(noise_np * 5 + 0.5)  # amplify for visibility
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Adversarial\nClass: {adv_idx}")
    plt.imshow(adv_img_np)
    plt.axis('off')

    print(f"\nOriginal Prediction: {orig_idx}")
    print(f"Adversarial Prediction: {adv_idx}")

    plt.show()


if __name__ == "__main__":
    run_experiment("sample.jpg", epsilon=0.02)