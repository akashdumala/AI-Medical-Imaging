
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(model, input_tensor, target_layer, target_index=None):
    """
    Generate a Grad-CAM overlay for a single image tensor.
    This visualizes model sensitivity, not a validated lesion boundary.
    """
    model.eval()
    activations = []

    def capture_activation(module, inputs, output):
        activations.append(output)

    hook = target_layer.register_forward_hook(capture_activation)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients are required for Grad-CAM.
        with torch.enable_grad():
            input_tensor = input_tensor.detach().requires_grad_(True)
            logits = model(input_tensor)

            if target_index is None:
                target_index = int(logits.argmax(dim=1).item())

            score = logits[0, target_index]

            gradients = torch.autograd.grad(
                outputs=score,
                inputs=activations[0],
                retain_graph=False,
                create_graph=False,
            )[0]

            feature_maps = activations[0].detach()

            # Average gradients across spatial dimensions.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted sum of feature maps.
            cam = (weights * feature_maps).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            cam_min = cam.min()
            cam_max = cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach().cpu()

    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def create_overlay(original_image, cam):
    """Blend a Grad-CAM heatmap with the original image."""
    original = original_image.convert("RGB").resize((224, 224))

    values = cam.flatten().tolist()
    heatmap = Image.new("RGB", (224, 224))

    # Red indicates higher activation; blue indicates lower activation.
    heatmap.putdata([
        (int(v * 255), 0, int((1.0 - v) * 100))
        for v in values
    ])

    overlay = Image.blend(original, heatmap, alpha=0.4)

    output = BytesIO()
    overlay.save(output, format="PNG")
    output.seek(0)

    return output
