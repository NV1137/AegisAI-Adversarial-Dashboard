import torch
import torch.nn.functional as F

def fgsm_attack(model, image, epsilon):
    image.requires_grad = True

    output = model(image)
    label = output.argmax(dim=1)

    loss = F.cross_entropy(output, label)

    model.zero_grad()
    loss.backward()

    data_grad = image.grad.data
    perturbed_image = image + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    noise = perturbed_image - image

    return perturbed_image.detach(), noise.detach()