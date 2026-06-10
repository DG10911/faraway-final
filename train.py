"""
RailGuard-FSL++ Training Script

Usage:
    python train.py --data-dir ./data/mendeley --output ./models
    python train.py --mode few-shot --n-shots 5 --n-ways 5 --n-episodes 100
"""

import argparse
import json
import numpy as np
from pathlib import Path

from backend.pipeline.phase1_input_validation import InputValidator
from backend.pipeline.phase2_rail_extraction import RailExtractor
from backend.pipeline.phase3_patch_pipeline import PatchPipeline
from backend.models.dinov2_encoder import DINOv2Encoder
from backend.models.patchcore_anomaly import PatchCoreDetector
from backend.models.prototypical_network import evaluate_few_shot
from backend.datasets.dataset_loader import DatasetLoader, DEFECT_CLASSES
from backend.utils.embeddings_db import EmbeddingDatabase


def main():
    parser = argparse.ArgumentParser(description="RailGuard-FSL++ Training")
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    parser.add_argument("--mode", choices=["healthy", "embed", "few-shot", "benchmark"], default="healthy")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n-ways", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    print(f"RailGuard-FSL++ | Device: {device} | Mode: {args.mode}")

    validator = InputValidator()
    extractor = RailExtractor()
    patcher = PatchPipeline(patch_size=256, overlap=0.25)
    encoder = DINOv2Encoder(device=device)
    db = EmbeddingDatabase()

    if args.mode == "healthy" and args.data_dir:
        loader = DatasetLoader()
        data = loader.load_mendeley(args.data_dir)
        healthy_imgs = data.get("healthy", [])
        print(f"Found {len(healthy_imgs)} healthy images")

        all_embeddings = []
        for i, img in enumerate(healthy_imgs):
            validation = validator.validate(img)
            if validation["status"] != "valid":
                print(f"  Skipping img {i}: {validation['reason']}")
                continue
            rail = extractor.extract_rail_region(img)
            if rail is None:
                continue
            patches = patcher.extract_patches(rail)
            patch_imgs = [p for p, _ in patches]
            embs = encoder.embed_patches(patch_imgs)
            all_embeddings.append(embs)
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(healthy_imgs)}")

        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            db.store("healthy", all_embeddings)
            detector = PatchCoreDetector(device=device)
            detector.fit(all_embeddings)
            print(f"Trained PatchCore on {len(all_embeddings)} healthy embeddings")
            print(f"Anomaly threshold: {detector.anomaly_threshold:.4f}")
            torch.save({"embeddings": all_embeddings, "threshold": detector.anomaly_threshold}, output_dir / "patchcore_memory.pt")

    elif args.mode == "embed" and args.data_dir:
        loader = DatasetLoader()
        data = loader.load_mendeley(args.data_dir)
        for cls_name, images in data.items():
            class_embs = []
            for img in images:
                rail = extractor.extract_rail_region(img)
                if rail is not None:
                    patches = patcher.extract_patches(rail)
                    patch_imgs = [p for p, _ in patches]
                    embs = encoder.embed_patches(patch_imgs)
                    class_embs.append(embs.mean(axis=0))
            if class_embs:
                combined = np.array(class_embs)
                db.store(cls_name, combined)
                print(f"Stored {len(combined)} embeddings for '{cls_name}'")

    elif args.mode == "few-shot":
        all_embs = db.get_all()
        if not all_embs:
            print("No embeddings in database. Run 'embed' mode first.")
            return
        print(f"Evaluating {args.n_ways}-way {args.n_shots}-shot over {args.n_episodes} episodes...")
        result = evaluate_few_shot(all_embs, args.n_ways, args.n_shots, args.n_episodes)
        print(json.dumps(result, indent=2))
        with open(output_dir / "few_shot_eval.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "benchmark":
        comparison = {
            "Baseline CNN": {"accuracy": None, "note": "Train from scratch on target domain"},
            "Transfer Learning": {"accuracy": None, "note": "ImageNet pretrain → fine-tune on target"},
            "RailGuard-FSL++": {"accuracy": None, "note": "DINOv2 + Prototypical Networks (ours)"},
            "protocol": "Train on Mendeley. Adapt with 5 support images from RSDDs.",
        }
        print(json.dumps(comparison, indent=2))
        with open(output_dir / "ablation_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    print("Done.")


import torch

if __name__ == "__main__":
    main()
