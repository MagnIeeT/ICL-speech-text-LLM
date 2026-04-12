import json
import os
import traceback

from config.data_config.master_config import DatasetType
from utils.evaluation_utils import evaluate_predictions


def reprocess_results(results_file: str):
    """Reprocess a results file and write per-dataset metrics for active datasets."""
    try:
        dir_path = os.path.dirname(results_file)
        base_name = os.path.basename(results_file)

        if "metrics" in base_name:
            results_file = results_file.replace("_metrics.json", "_results.json")
            base_name = os.path.basename(results_file)

        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file not found: {results_file}")

        with open(results_file, "r") as handle:
            raw_results = json.load(handle)

        if isinstance(raw_results, dict):
            results = [raw_results[key] for key in raw_results.keys() if isinstance(raw_results[key], dict)]
        else:
            results = raw_results

        if not results:
            raise ValueError("No valid predictions found in file")

        dataset_types = {
            "voxceleb": DatasetType.VOXCELEB,
            "hvb": DatasetType.HVB,
            "voxpopuli": DatasetType.VOXPOPULI,
            "meld_emotion": DatasetType.MELD_EMOTION,
        }

        combined_metrics = {}
        for dataset_name, dataset_type in dataset_types.items():
            dataset_results = [r for r in results if str(r.get("dataset_type", "")).lower() == dataset_name]
            if dataset_results:
                combined_metrics[dataset_name] = evaluate_predictions(dataset_results, dataset_type)

        output_file = os.path.join(dir_path, base_name.replace("_results.json", "_metrics_combined.json"))
        with open(output_file, "w") as handle:
            json.dump(combined_metrics, handle, indent=2)

        print(f"Saved combined metrics to {output_file}")

    except Exception as exc:
        print(f"Error processing file: {exc}")
        traceback.print_exc()


def process_folder(folder_path: str):
    """Process all *_results.json files recursively under folder_path."""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith("_results.json"):
                reprocess_results(os.path.join(root, file_name))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process results files in a folder")
    parser.add_argument("folder_path", type=str, help="Path to folder containing results files")
    args = parser.parse_args()

    if os.path.exists(args.folder_path):
        process_folder(args.folder_path)
    else:
        print(f"Folder not found: {args.folder_path}")
