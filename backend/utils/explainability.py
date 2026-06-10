import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class ExplainabilityEngine:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate_gradcam(
        self,
        model: torch.nn.Module,
        image: np.ndarray,
        target_layer: Optional[str] = None,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        model.eval()
        tensor = self._preprocess_image(image).to(self.device)
        tensor.requires_grad_()
        output = model(tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        model.zero_grad()
        output[0, class_idx].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(tensor).detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap_np = heatmap.cpu().numpy()
        heatmap_np = cv2.resize(heatmap_np, (image.shape[1], image.shape[0]))
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        heatmap_np = np.uint8(255 * heatmap_np)
        return heatmap_np

    @staticmethod
    def generate_attention_overlay(image: np.ndarray, attention_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        attention_colored = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 1 - alpha, attention_colored, alpha, 0)
        return overlay

    @staticmethod
    def generate_anomaly_heatmap(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        anomaly_norm = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(anomaly_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        return tensor
