import albumentations as A
import cv2
import numpy as np
from typing import List, Callable


def get_training_augmentations() -> A.Compose:
    return A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.8),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.MotionBlur(blur_limit=(3, 7), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_test_augmentations() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def augment_image(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    return pipeline(image=image)["image"]


def generate_synthetic_bright_streaks(image: np.ndarray, n_streaks: int = 3) -> np.ndarray:
    result = image.copy()
    h, w = image.shape[:2]
    for _ in range(n_streaks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        angle = np.random.uniform(-30, 30)
        length = np.random.randint(20, 80)
        thickness = np.random.randint(2, 8)
        brightness = np.random.randint(200, 255)
        x2 = int(x + length * np.cos(np.radians(angle)))
        y2 = int(y + length * np.sin(np.radians(angle)))
        cv2.line(result, (x, y), (x2, y2), (brightness, brightness, brightness), thickness)
    return result
