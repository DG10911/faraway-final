import cv2
import numpy as np
from typing import List, Tuple


class PatchPipeline:
    def __init__(self, patch_size: int = 256, overlap: float = 0.25):
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))

    def extract_patches(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        h, w = image.shape[:2]
        patches = []
        y = 0
        while y + self.patch_size <= h:
            x = 0
            while x + self.patch_size <= w:
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                coords = (x, y, x + self.patch_size, y + self.patch_size)
                patches.append((patch, coords))
                x += self.stride
            y += self.stride
        if not patches:
            resized = cv2.resize(image, (self.patch_size, self.patch_size))
            patches.append((resized, (0, 0, self.patch_size, self.patch_size)))
        return patches

    def reconstruct_from_patches(self, patches: List[Tuple[np.ndarray, np.ndarray]], original_shape: Tuple[int, int]) -> np.ndarray:
        h, w = original_shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        for patch_data, coords in patches:
            x1, y1, x2, y2 = coords
            if patch_data.shape != (self.patch_size, self.patch_size):
                heatmap[y1:y2, x1:x2] += cv2.resize(patch_data, (self.patch_size, self.patch_size))
            else:
                heatmap[y1:y2, x1:x2] += patch_data
            count_map[y1:y2, x1:x2] += 1
        count_map = np.maximum(count_map, 1e-6)
        return heatmap / count_map
