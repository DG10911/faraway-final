"""
RailGuard-FSL++ Experiment Runner

Runs the full research suite and writes results/experiment_results.json:
  1. Healthy-only PatchCore training (DINOv2 patch tokens)
  2. Per-image class embeddings (top-K anomalous patches)
  3. Few-shot episodic evaluation (closed-set + open-set, mean ± std, 95% CI)
  4. Conformal calibration -> guaranteed-recall operating point
  5. Cross-domain adaptation (optional, with --rsdds-dir / --fastener-dir)
  6. Ablation vs transfer-learning baselines (--ablation)

    python run_experiments.py --data-dir ./data/mendeley --ablation
"""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.pipeline.orchestrator import RailGuardFSL
from backend.datasets.dataset_loader import DatasetLoader
from backend.evaluation.cross_domain import CrossDomainEvaluator
from backend.utils.conformal import conformal_recall_threshold, evaluate_operating_point

TOP_K_PATCHES = 3


def main():
    parser = argparse.ArgumentParser(description="RailGuard-FSL++ Experiments")
    parser.add_argument("--data-dir", type=str, required=True, help="Mendeley dataset path")
    parser.add_argument("--rsdds-dir", type=str, help="RSDDs dataset path (optional)")
    parser.add_argument("--fastener-dir", type=str, help="Fastener dataset path (optional)")
    parser.add_argument("--output", type=str, default="results", help="Output dir")
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n-ways", type=int, default=5)
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--ablation", action="store_true", help="Run baseline ablation (slower)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RailGuard-FSL++ Experiment Suite")
    print("=" * 60)

    system = RailGuardFSL()
    print(f"Device: {system.device}")
    loader = DatasetLoader()
    results = {}

    print("\n[1/6] Loading Mendeley dataset (with perceptual-hash dedup)...")
    raw = loader.load_mendeley(args.data_dir)
    mendeley_data = {cls: loader.deduplicate(imgs) for cls, imgs in raw.items() if imgs}
    for cls, imgs in mendeley_data.items():
        print(f"  {cls}: {len(imgs)} images")

    print("[2/6] Training PatchCore on healthy patch tokens...")
    healthy_tokens = []
    for img in mendeley_data.get("healthy", []):
        features = system.embed_image(img)
        if features is None:
            continue
        healthy_tokens.append(features["tokens"].reshape(-1, features["tokens"].shape[-1]))
    if not healthy_tokens:
        print("ERROR: no usable healthy images — check the dataset layout (data/mendeley/healthy/*.jpg)")
        return
    tokens = np.concatenate(healthy_tokens, axis=0)
    system.initialize(tokens, threshold_percentile=95.0)
    system.embedding_db.store("healthy_tokens", tokens)
    results["patchcore"] = {
        "n_healthy_tokens": int(len(tokens)),
        "memory_bank_size": int(len(system.anomaly_detector.memory.memory_bank)),
        "threshold": float(system.anomaly_detector.anomaly_threshold),
    }
    print(f"  Memory bank: {results['patchcore']['memory_bank_size']} vectors, threshold {results['patchcore']['threshold']:.4f}")

    print("[3/6] Per-image class embeddings + image-level anomaly scores...")
    image_scores = {}  # class -> list of image-level anomaly scores
    for cls_name, images in mendeley_data.items():
        class_embs, class_scores = [], []
        for img in images:
            features = system.embed_image(img)
            if features is None:
                continue
            flat = features["tokens"].reshape(-1, features["tokens"].shape[-1])
            scores = system.anomaly_detector.score_features(flat)
            class_scores.append(float(scores.max()))
            if cls_name == "healthy":
                class_embs.append(features["cls"].mean(axis=0))
            else:
                per_patch = scores.reshape(len(features["patch_coords"]), -1).max(axis=1)
                top = np.argsort(per_patch)[-min(TOP_K_PATCHES, len(per_patch)):]
                class_embs.append(features["cls"][top].mean(axis=0))
        if class_embs:
            system.embedding_db.store(cls_name, np.array(class_embs))
            image_scores[cls_name] = class_scores
            print(f"  {cls_name}: {len(class_embs)} embeddings")

    print(f"[4/6] Few-shot evaluation ({args.n_ways}-way {args.n_shots}-shot, {args.n_episodes} episodes)...")
    all_embs = {k: v for k, v in system.embedding_db.get_all().items() if k != "healthy_tokens"}
    closed = system.evaluate_few_shot_performance(all_embs, args.n_ways, args.n_shots, args.n_episodes)
    open_set = system.evaluate_few_shot_performance(all_embs, args.n_ways, args.n_shots, args.n_episodes, open_set=True)
    results["few_shot"] = {"closed_set": closed, "open_set": open_set}
    print(f"  Closed-set: {closed['mean_accuracy']:.4f} ± {closed['std_accuracy']:.4f} (CI95 {closed['ci_95']:.4f})")
    print(f"  Open-set:   acc {open_set['mean_accuracy']:.4f}, unknown detection {open_set.get('open_set_detection_rate', 0):.4f}")

    print(f"[5/6] Conformal calibration (target recall {args.target_recall})...")
    defect_scores = np.array([s for cls, sc in image_scores.items() if cls != "healthy" for s in sc])
    healthy_scores = np.array(image_scores.get("healthy", []))
    if len(defect_scores) >= 4:
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(defect_scores))
        half = len(defect_scores) // 2
        calib, test = defect_scores[perm[:half]], defect_scores[perm[half:]]
        conformal = conformal_recall_threshold(calib, args.target_recall)
        operating = evaluate_operating_point(healthy_scores, test, conformal["threshold"])
        results["conformal"] = {"calibration": conformal, "held_out_operating_point": operating}
        print(f"  Guaranteed recall >= {conformal['guaranteed_recall']}: threshold {conformal['threshold']:.4f}")
        print(f"  Held-out: recall {operating['empirical_recall']:.4f}, FPR {operating['false_positive_rate']:.4f}")
    else:
        print("  Skipped (need >= 4 defect images)")

    print("[6/6] Cross-domain adaptation...")
    target_data = {}
    if args.rsdds_dir:
        target_data.update({f"rsdds_{k}": loader.deduplicate(v) for k, v in loader.load_rsdds(args.rsdds_dir).items() if v})
    if args.fastener_dir:
        target_data.update({k: loader.deduplicate(v) for k, v in loader.load_fastener(args.fastener_dir).items() if v})
    if target_data:
        cross_domain = CrossDomainEvaluator(device=system.device, encoder=system.encoder)
        cd_result = cross_domain.evaluate_adaptation(mendeley_data, target_data, args.n_shots, args.n_episodes)
        results["cross_domain"] = cd_result
        print(f"  Source accuracy: {cd_result['source_domain']['mean_accuracy']:.4f}")
        print(f"  Target accuracy: {cd_result['target_domain']['mean_accuracy']:.4f}")
        print(f"  Domain gap:      {cd_result['domain_gap']['accuracy_drop']:.4f}")
    else:
        print("  Skipped (pass --rsdds-dir / --fastener-dir to enable)")

    if args.ablation:
        print("[+] Ablation vs baselines...")
        from backend.evaluation.benchmark import run_ablation
        crops = {}
        for cls_name, images in mendeley_data.items():
            regions = [system.extractor.extract_rail_region(img) for img in images]
            regions = [r for r in regions if r is not None]
            if regions:
                crops[cls_name] = regions
        results["ablation"] = run_ablation(crops, all_embs, args.n_ways, args.n_shots, args.n_episodes, device=system.device)
        for name, row in results["ablation"].items():
            if isinstance(row, dict) and "mean_accuracy" in row:
                print(f"  {name}: {row['mean_accuracy']:.4f} ± {row['std_accuracy']:.4f}")

    output_path = output_dir / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
