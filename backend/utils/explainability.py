import cv2
import numpy as np
from typing import Optional


class ExplainabilityEngine:
    """Visual explanations for detections.

    - Anomaly heatmap: PatchCore distance map over the rail region.
    - Attention map: DINOv2 CLS-to-patch self-attention (recomputed from the
      qkv projection, see DINOv2Encoder.get_cls_attention) — gradient-free and
      faithful to what the ViT actually attends to.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device

    @staticmethod
    def generate_attention_map(encoder, image: np.ndarray) -> np.ndarray:
        """Overlay the encoder's CLS attention on the image. Returns RGB uint8."""
        attn = encoder.get_cls_attention(image)  # (grid, grid) in [0, 1]
        attn_resized = cv2.resize(attn, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        attn_u8 = np.uint8(255 * np.clip(attn_resized, 0, 1))
        return ExplainabilityEngine._overlay(image, attn_u8, alpha=0.5)

    @staticmethod
    def generate_anomaly_heatmap(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        anomaly_norm = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        return ExplainabilityEngine._overlay(image, anomaly_norm, alpha=alpha)

    @staticmethod
    def _overlay(image: np.ndarray, intensity_u8: np.ndarray, alpha: float) -> np.ndarray:
        colored = cv2.applyColorMap(intensity_u8, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)  # keep everything RGB internally
        if image.dtype != np.uint8:
            image = np.clip(image * 255 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
