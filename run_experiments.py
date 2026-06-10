"""
RailGuard-FSL++ Experiment Runner

Run all research experiments:
    python run_experiments.py --data-dir ./data/mendeley
"""

import argparse
import json
import numpy as np
from pathlib import Path

from backend.pipeline.orchestrator import RailGuardFSL
from backend.datasets.dataset_loader import DatasetLoader
from backend.evaluation.metrics import Evaluator
from backend.evaluation.cross_domain import CrossDomainEvaluator
from backend.utils.embeddings_db import EmbeddingDatabase


def main():
    parser = argparse.ArgumentParser(description="RailGuard-FSL++ Experiments")
    parser.add_argument("--data-dir", type=str, required=True, help="Mendeley dataset path")
    parser.add_argument("--rsdds-dir", type=str, help="RSDDs dataset path (optional)")
    parser.add_argument("--fastener-dir", type=str, help="Fastener dataset path (optional)")
    parser.add_argument("--output", type=str, default="results", help="Output dir")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n-ways", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    print("=" * 60)
    print("RailGuard-FSL++ Experiment Suite")
    print("=" * 60)
    print(f"Device: {device}")

    system = RailGuardFSL(device=device)
    loader = DatasetLoader()
    results = {}

    # Phase 1: Load and embed Mendeley dataset
    print("\n[1/5] Loading Mendeley dataset...")
    mendeley_data = loader.load_mendeley(args.data_dir)
    print(f"  Classes: {[k for k, v in mendeley_data.items() if len(v) > 0]}")

    # Embed all classes
    print("[2/5] Generating DINOv2 embeddings...")
    for cls_name, images in mendeley_data.items():
        if not images:
            continue
        print(f"  Embedding {cls_name} ({len(images)} images)...")
        for img in images:
            rail = system.extractor.extract_rail_region(img)
            if rail is not None:
                patches = system.patcher.extract_patches(rail)
                patch_imgs = [p for p, _ in patches]
                embs = system.encoder.embed_patches(patch_imgs)
                system.embedding_db.store(cls_name, embs)

    # Phase 2: Train PatchCore on healthy only
    print("[3/5] Training PatchCore anomaly detector...")
    healthy_embs = system.embedding_db.get("healthy")
    if healthy_embs is not None:
        system.initialize(healthy_embs, threshold_percentile=95.0)
        results["patchcore_threshold"] = system.anomaly_detector.anomaly_threshold

    # Phase 3: Few-shot evaluation
    print(f"[4/5] Few-shot evaluation ({args.n_ways}-way {args.n_shots}-shot)...")
    all_embs = system.embedding_db.get_all()
    few_shot_result = system.evaluate_few_shot_performance(
        all_embs, args.n_ways, args.n_shots, args.n_episodes
    )
    results["few_shot"] = few_shot_result
    print(f"  Mean accuracy: {few_shot_result['mean_accuracy']:.4f} ± {few_shot_result['std_accuracy']:.4f}")

    # Phase 4: Cross-domain adaptation
    print("[5/5] Cross-domain adaptation...")
    cross_domain = CrossDomainEvaluator(device=device)
    target_data = {}
    if args.rsdds_dir:
        rsdds_data = loader.load_rsdds(args.rsdds_dir)
        target_data.update(rsdds_data)
    if args.fastener_dir:
        fastener_data = loader.load_fastener(args.fastener_dir)
        target_data.update(fastener_data)

    if target_data:
        cd_result = cross_domain.evaluate_adaptation(mendeley_data, target_data, args.n_shots, args.n_episodes)
        results["cross_domain"] = cd_result
        print(f"  Source accuracy: {cd_result['source_domain']['mean_accuracy']:.4f}")
        print(f"  Target accuracy: {cd_result['target_domain']['mean_accuracy']:.4f}")
        domain_gap = cd_result["source_domain"]["mean_accuracy"] - cd_result["target_domain"]["mean_accuracy"]
        print(f"  Domain gap: {domain_gap:.4f}")

    # Save results
    output_path = output_dir / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
