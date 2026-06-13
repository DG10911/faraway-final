import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 backbone producing L2-normalized CLS and patch-token features.

    CLS embeddings feed the few-shot prototypical classifier; patch tokens feed
    the PatchCore memory bank (AnomalyDINO-style dense scoring). Inputs are
    resized to INPUT_SIZE, which must be divisible by the ViT patch size (14).
    """

    MODEL_NAMES = {"vits14", "vitb14", "vitl14", "vitg14", "vits14_reg", "vitb14_reg", "vitl14_reg", "vitg14_reg"}
    INPUT_SIZE = 224  # 224 = 16 * 14 -> 16x16 token grid
    PATCH_SIZE = 14

    def __init__(self, model_name: str = "vits14", device: Optional[str] = None, normalize: bool = True):
        super().__init__()
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"model_name must be one of {self.MODEL_NAMES}, got '{model_name}'")
        from ..utils.device import resolve_device
        self.device = resolve_device(device)
        self.normalize = normalize
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_name}")
        self.model.eval()
        self.model.to(self.device)
        self.embed_dim = self.model.embed_dim
        self.grid_size = self.INPUT_SIZE // self.PATCH_SIZE
        self.num_register_tokens = getattr(self.model, "num_register_tokens", 0)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        with torch.no_grad():
            cls_token = self.model(x)
        if self.normalize:
            cls_token = torch.nn.functional.normalize(cls_token, dim=-1)
        return cls_token

    def embed_patches(self, patches: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """CLS embedding per patch image. Returns (n_patches, embed_dim)."""
        embeddings = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batch_tensor = torch.stack([self._preprocess(p) for p in batch]).to(self.device)
            emb = self.forward(batch_tensor)
            embeddings.append(emb.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def embed_patch_tokens(self, patches: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Dense patch-token features per patch image.

        Returns (n_patches, grid_size * grid_size, embed_dim), L2-normalized.
        """
        all_tokens = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            batch_tensor = torch.stack([self._preprocess(p) for p in batch]).to(self.device)
            batch_tensor = (batch_tensor - self.mean) / self.std
            with torch.no_grad():
                features = self.model.forward_features(batch_tensor)
            tokens = features["x_norm_patchtokens"]
            if self.normalize:
                tokens = torch.nn.functional.normalize(tokens, dim=-1)
            all_tokens.append(tokens.cpu().numpy())
        return np.concatenate(all_tokens, axis=0)

    def get_cls_attention(self, image: np.ndarray) -> np.ndarray:
        """CLS-to-patch attention of the last block, as a (grid, grid) map in [0, 1].

        Captures the qkv projection output via a forward hook and recomputes the
        CLS attention row explicitly, which works regardless of whether the
        model uses memory-efficient attention internally.
        """
        captured = {}

        def hook(_module, _inputs, output):
            captured["qkv"] = output.detach()

        handle = self.model.blocks[-1].attn.qkv.register_forward_hook(hook)
        try:
            tensor = self._preprocess(image).unsqueeze(0).to(self.device)
            tensor = (tensor - self.mean) / self.std
            with torch.no_grad():
                self.model.forward_features(tensor)
        finally:
            handle.remove()

        qkv = captured["qkv"]  # (1, n_tokens, 3 * dim)
        num_heads = self.model.blocks[-1].attn.num_heads
        b, n, _ = qkv.shape
        head_dim = self.embed_dim // num_heads
        qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]  # (1, heads, n_tokens, head_dim)
        attn = (q[:, :, 0:1] @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # CLS row
        attn = attn.softmax(dim=-1).mean(dim=1).squeeze()  # (n_tokens,)
        offset = 1 + self.num_register_tokens  # skip CLS (+ registers)
        patch_attn = attn[offset:].reshape(self.grid_size, self.grid_size).cpu().numpy()
        patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)
        return patch_attn

    def _preprocess(self, patch: np.ndarray) -> torch.Tensor:
        if patch.dtype != np.float32:
            patch = patch.astype(np.float32) / 255.0
        if patch.ndim == 2:
            patch = np.stack([patch] * 3, axis=-1)
        elif patch.shape[2] == 4:
            patch = patch[:, :, :3]
        if patch.shape[:2] != (self.INPUT_SIZE, self.INPUT_SIZE):
            patch = cv2.resize(patch, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        return tensor
