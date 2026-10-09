import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.evaluation.evaluation_dataset import DEFAULT_EVALUATION_DATASET
from src.evaluation.retrieval_metrics import evaluate_retrieval_benchmark
from src.retrieval.hybrid_search import HybridSearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval precision, recall, MRR, and nDCG.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff rank K for evaluation metrics")
    args = parser.parse_args()

    hybrid_engine = HybridSearchEngine()

    def retriever_fn(query: str, top_k: int):
        return hybrid_engine.search(query=query, top_k=top_k)

    print("\nRunning benchmark evaluation over benchmark query set...")
    metrics = evaluate_retrieval_benchmark(DEFAULT_EVALUATION_DATASET, retriever_fn, k=args.k)

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    for metric_name, val in metrics.items():
        print(f"{metric_name:<30}: {val}")
    print("=" * 60)


if __name__ == "__main__":
    main()
