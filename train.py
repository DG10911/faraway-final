"""
RailGuard-FSL++ Training Script

Usage:
    python train.py --data-dir ./data/mendeley --mode healthy     # stage 1: healthy memory bank
    python train.py --data-dir ./data/mendeley --mode embed       # per-image class embeddings
    python train.py --mode few-shot --n-shots 5 --n-ways 5 --n-episodes 100
    python train.py --data-dir ./data/mendeley --mode benchmark   # real ablation table
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.pipeline.phase1_input_validation import InputValidator
from backend.pipeline.phase2_rail_extraction import RailExtractor
from backend.pipeline.phase3_patch_pipeline import PatchPipeline
from backend.models.dinov2_encoder import DINOv2Encoder
from backend.models.patchcore_anomaly import PatchCoreDetector
from backend.models.prototypical_network import evaluate_few_shot
from backend.datasets.dataset_loader import DatasetLoader
from backend.utils.embeddings_db import EmbeddingDatabase

TOP_K_PATCHES = 3  # most-anomalous patches kept per image, so tiny defects aren't averaged away


def get_device(arg_device):
    from backend.utils.device import resolve_device
    return resolve_device(arg_device)


def rail_crops_by_class(data, extractor, deduplicate):
    crops = {}
    for cls_name, images in data.items():
        if not images:
            continue
        images = deduplicate(images)
        regions = [extractor.extract_rail_region(img) for img in images]
        regions = [r for r in regions if r is not None]
        if regions:
            crops[cls_name] = regions
    return crops


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
    device = get_device(args.device)
    print(f"RailGuard-FSL++ | Device: {device} | Mode: {args.mode}")

    validator = InputValidator()
    extractor = RailExtractor()
    patcher = PatchPipeline(patch_size=256, overlap=0.25)
    db = EmbeddingDatabase()
    loader = DatasetLoader()

    if args.mode == "healthy":
        if not args.data_dir:
            parser.error("--data-dir is required for healthy mode")
        encoder = DINOv2Encoder(device=device)
        data = loader.load_mendeley(args.data_dir)
        healthy_imgs = loader.deduplicate(data.get("healthy", []))
        print(f"Found {len(healthy_imgs)} healthy images (after dedup)")

        all_tokens, all_cls = [], []
        for i, img in enumerate(healthy_imgs):
            validation = validator.validate(img)
            if validation["status"] != "valid":
                print(f"  Skipping img {i}: {validation['reason']}")
                continue
            rail = extractor.extract_rail_region(img)
            if rail is None:
                continue
            patch_imgs = [p for p, _ in patcher.extract_patches(rail)]
            tokens = encoder.embed_patch_tokens(patch_imgs)
            all_tokens.append(tokens.reshape(-1, tokens.shape[-1]))
            all_cls.append(encoder.embed_patches(patch_imgs))
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(healthy_imgs)}")

        if not all_tokens:
            print("No valid healthy images found.")
            return
        tokens = np.concatenate(all_tokens, axis=0)
        cls = np.concatenate(all_cls, axis=0)
        db.store("healthy_tokens", tokens)
        db.store("healthy", cls)
        detector = PatchCoreDetector(device=device)
        detector.fit(tokens)
        np.savez(
            output_dir / "patchcore_memory.npz",
            memory=detector.memory.memory_bank,
            threshold=detector.anomaly_threshold,
            healthy_scores=detector.healthy_scores,
        )
        print(f"Trained PatchCore on {len(tokens)} healthy token embeddings")
        print(f"Anomaly threshold (95th pct of healthy 1-NN distances): {detector.anomaly_threshold:.4f}")

    elif args.mode == "embed":
        if not args.data_dir:
            parser.error("--data-dir is required for embed mode")
        encoder = DINOv2Encoder(device=device)
        data = loader.load_mendeley(args.data_dir)

        detector = None
        memory_path = output_dir / "patchcore_memory.npz"
        healthy_tokens = db.get("healthy_tokens")
        if healthy_tokens is not None:
            detector = PatchCoreDetector(device=device)
            detector.fit(healthy_tokens)
        elif memory_path.exists():
            saved = np.load(memory_path)
            detector = PatchCoreDetector(device=device)
            detector.fit(saved["memory"])
        if detector is None:
            print("NOTE: no healthy memory bank found — falling back to mean patch embedding.")
            print("      Run --mode healthy first for defect-focused (top-K anomalous patch) embeddings.")

        for cls_name, images in data.items():
            images = loader.deduplicate(images)
            class_embs = []
            for img in images:
                rail = extractor.extract_rail_region(img)
                if rail is None:
                    continue
                patch_imgs = [p for p, _ in patcher.extract_patches(rail)]
                cls_embs = encoder.embed_patches(patch_imgs)
                if detector is not None and cls_name != "healthy":
                    tokens = encoder.embed_patch_tokens(patch_imgs)
                    scores = detector.score_features(tokens.reshape(-1, tokens.shape[-1]))
                    per_patch = scores.reshape(len(patch_imgs), -1).max(axis=1)
                    top = np.argsort(per_patch)[-min(TOP_K_PATCHES, len(patch_imgs)):]
                    class_embs.append(cls_embs[top].mean(axis=0))
                else:
                    class_embs.append(cls_embs.mean(axis=0))
            if class_embs:
                db.store(cls_name, np.array(class_embs))  # one embedding per image -> no patch-level leakage
                print(f"Stored {len(class_embs)} per-image embeddings for '{cls_name}'")

    elif args.mode == "few-shot":
        all_embs = {k: v for k, v in db.get_all().items() if k != "healthy_tokens"}
        if not all_embs:
            print("No embeddings in database. Run 'embed' mode first.")
            return
        print(f"Evaluating {args.n_ways}-way {args.n_shots}-shot over {args.n_episodes} episodes...")
        closed = evaluate_few_shot(all_embs, args.n_ways, args.n_shots, args.n_episodes)
        open_set = evaluate_few_shot(all_embs, args.n_ways, args.n_shots, args.n_episodes, open_set=True)
        result = {"closed_set": closed, "open_set": open_set}
        print(json.dumps(result, indent=2))
        with open(output_dir / "few_shot_eval.json", "w") as f:
            json.dump(result, f, indent=2)

    elif args.mode == "benchmark":
        if not args.data_dir:
            parser.error("--data-dir is required for benchmark mode")
        from backend.evaluation.benchmark import run_ablation

        data = loader.load_mendeley(args.data_dir)
        crops = rail_crops_by_class(data, extractor, loader.deduplicate)
        dinov2_embs = {k: v for k, v in db.get_all().items() if k != "healthy_tokens"}
        if not dinov2_embs:
            print("No DINOv2 embeddings in database. Run 'embed' mode first.")
            return
        print("Running ablation (all rows use the same rail crops + episodic protocol)...")
        comparison = run_ablation(crops, dinov2_embs, args.n_ways, args.n_shots, args.n_episodes, device=device)
        print(json.dumps(comparison, indent=2))
        with open(output_dir / "ablation_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
