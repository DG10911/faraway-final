import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class DINOv2Encoder(nn.Module):
    def __init__(self, model_name: str = "vit_small", device: Optional[str] = None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_name}")
        self.model.eval()
        self.model.to(self.device)
        self.embed_dim = self.model.embed_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        with torch.no_grad():
            cls_token = self.model(x)
        return cls_token

    def embed_patches(self, patches: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batch_tensor = torch.stack([self._preprocess(p) for p in batch]).to(self.device)
            emb = self.forward(batch_tensor)
            embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def _preprocess(self, patch: np.ndarray) -> torch.Tensor:
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32) / 255.0
        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)
        elif patch.shape[2] == 4:
            patch = patch[:, :, :3]
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        return tensor.unsqueeze(0)
