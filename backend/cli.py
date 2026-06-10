import argparse
import json
import numpy as np
from pathlib import Path

from .pipeline.orchestrator import RailGuardFSL
from .datasets.dataset_loader import DatasetLoader, DEFECT_CLASSES
from .evaluation.benchmark import BenchmarkRunner, create_ablation_comparison


def main():
    parser = argparse.ArgumentParser(description="RailGuard-FSL++ CLI")
    parser.add_argument("--mode", choices=["train", "detect", "evaluate", "embed", "benchmark"], required=True)
    parser.add_argument("--input", type=str, help="Input path (image, dir, or dataset)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--n-shots", type=int, default=5, help="Number of support shots")
    parser.add_argument("--n-ways", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-episodes", type=int, default=100, help="Evaluation episodes")
    args = parser.parse_args()

    system = RailGuardFSL()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        loader = DatasetLoader()
        data = loader.load_mendeley(args.input) if args.input else {}
        healthy_imgs = data.get("healthy", [])
        if healthy_imgs:
            all_embs = []
            for img in healthy_imgs:
                rail = system.extractor.extract_rail_region(img)
                if rail is not None:
                    patches = system.patcher.extract_patches(rail)
                    patch_imgs = [p for p, _ in patches]
                    embs = system.encoder.embed_patches(patch_imgs)
                    all_embs.append(embs)
            if all_embs:
                all_embs = np.concatenate(all_embs, axis=0)
                system.initialize(all_embs)
                system.embedding_db.store("healthy", all_embs)
                print(f"Trained on {len(healthy_imgs)} healthy images -> {len(all_embs)} embeddings")

    elif args.mode == "detect":
        img = cv2.imread(args.input)
        if img is None:
            print(f"Cannot read {args.input}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = system.process_frame(img)
        output_path = output_dir / "detection_result.json"
        serializable = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(json.dumps(serializable, indent=2))

    elif args.mode == "evaluate":
        result = system.evaluate_few_shot_performance(
            system.embedding_db.get_all(),
            args.n_ways, args.n_shots, args.n_episodes
        )
        output_path = output_dir / "few_shot_eval.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

    elif args.mode == "embed":
        loader = DatasetLoader()
        data = loader.load_mendeley(args.input) if args.input else {}
        for cls_name, images in data.items():
            all_embs = []
            for img in images:
                rail = system.extractor.extract_rail_region(img)
                if rail is not None:
                    patches = system.patcher.extract_patches(rail)
                    patch_imgs = [p for p, _ in patches]
                    embs = system.encoder.embed_patches(patch_imgs)
                    all_embs.append(embs)
            if all_embs:
                combined = np.concatenate(all_embs, axis=0)
                system.embedding_db.store(cls_name, combined)
                print(f"Stored {len(combined)} embeddings for class '{cls_name}'")

    elif args.mode == "benchmark":
        comparison = create_ablation_comparison()
        output_path = output_dir / "ablation_comparison.json"
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(json.dumps(comparison, indent=2))

    else:
        print("Invalid mode")


import cv2

if __name__ == "__main__":
    main()
