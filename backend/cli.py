import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .pipeline.orchestrator import RailGuardFSL
from .datasets.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(description="RailGuard-FSL++ CLI")
    parser.add_argument("--mode", choices=["train", "detect", "evaluate", "embed"], required=True)
    parser.add_argument("--input", type=str, help="Input path (image, dir, or dataset)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--n-shots", type=int, default=5, help="Number of support shots")
    parser.add_argument("--n-ways", type=int, default=5, help="Number of classes per episode")
    parser.add_argument("--n-episodes", type=int, default=100, help="Evaluation episodes")
    args = parser.parse_args()

    system = RailGuardFSL()
    loader = DatasetLoader()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        data = loader.load_mendeley(args.input) if args.input else {}
        healthy_imgs = loader.deduplicate(data.get("healthy", []))
        all_tokens = []
        for img in healthy_imgs:
            features = system.embed_image(img)
            if features is not None:
                all_tokens.append(features["tokens"].reshape(-1, features["tokens"].shape[-1]))
        if all_tokens:
            tokens = np.concatenate(all_tokens, axis=0)
            system.initialize(tokens)
            system.embedding_db.store("healthy_tokens", tokens)
            print(f"Trained on {len(healthy_imgs)} healthy images -> {len(tokens)} token embeddings")

    elif args.mode == "detect":
        img = cv2.imread(args.input)
        if img is None:
            print(f"Cannot read {args.input}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = system.process_frame(img)
        serializable = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        output_path = output_dir / "detection_result.json"
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(json.dumps(serializable, indent=2))

    elif args.mode == "evaluate":
        all_embs = {k: v for k, v in system.embedding_db.get_all().items() if k != "healthy_tokens"}
        result = system.evaluate_few_shot_performance(all_embs, args.n_ways, args.n_shots, args.n_episodes, open_set=True)
        output_path = output_dir / "few_shot_eval.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))

    elif args.mode == "embed":
        data = loader.load_mendeley(args.input) if args.input else {}
        for cls_name, images in data.items():
            images = loader.deduplicate(images)
            class_embs = []
            for img in images:
                features = system.embed_image(img)
                if features is not None:
                    class_embs.append(features["cls"].mean(axis=0))
            if class_embs:
                system.embedding_db.store(cls_name, np.array(class_embs))
                print(f"Stored {len(class_embs)} per-image embeddings for class '{cls_name}'")


if __name__ == "__main__":
    main()
