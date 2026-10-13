from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib

from src.data import TARGET_COLUMN, MODEL_FEATURE_COLUMNS, MODEL_INPUT_COLUMNS, load_dataset, validate_frame
from src.models import make_supervised_models
from src.registry import import_trained_model, promote_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and activate an AADS model artifact for the local inference API.")
    parser.add_argument("--max-rows", type=int, default=50_000, help="Maximum labelled rows used to fit the estimator. Temporal features still use the full input sequence (default: 50000).")
    parser.add_argument("--input", type=Path, default=ROOT / "compressed_data.csv.gz", help="Input CSV or gzip CSV path.")
    parser.add_argument("--model", choices=["Random Forest", "Extra Trees"], default="Random Forest")
    args = parser.parse_args()
    if args.max_rows < 100:
        parser.error("--max-rows must be at least 100")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    data, notes = validate_frame(load_dataset(args.input))
    if TARGET_COLUMN not in data:
        raise SystemExit("Dataset needs a recognized Label column for supervised training.")
    data = data[data[TARGET_COLUMN].isin([0, 1])].copy()
    if data[TARGET_COLUMN].nunique() < 2:
        raise SystemExit("Training data must contain both Label=0 and Label=1 records.")
    model = make_supervised_models(max_train_rows=args.max_rows)[args.model]
    # Feature engineering sees the full chronological stream; only estimator fitting is capped.
    model.fit(data[MODEL_INPUT_COLUMNS], data[TARGET_COLUMN].astype(int))
    # This training artifact is not a substitute for a held-out benchmark.
    reference_columns = [c for c in MODEL_INPUT_COLUMNS if c in data.columns]
    reference_frame = data[reference_columns].copy()
    if len(reference_frame) > 5000:
        reference_frame = reference_frame.iloc[::max(1, len(reference_frame) // 5000)].head(5000).copy()
    registered = import_trained_model(model, args.model, threshold=0.5, dataset_name=args.input.name,
                                      row_count=len(data), reference_frame=reference_frame)
    active = promote_candidate(registered["run_id"], registered["model_name"])
    print(f"Registered training run: {registered['run_id']}")
    print(f"Active model: {active['model_version']}")
    print(f"Training records: {len(data):,}; normal: {(data[TARGET_COLUMN] == 0).sum():,}; attack: {(data[TARGET_COLUMN] == 1).sum():,}")
    print("Raw model input columns: " + ", ".join(MODEL_INPUT_COLUMNS))
    print("Temporal features are computed from prior messages; absolute Time is not used as a direct predictor.")
    print("NOTE: this is a training artifact, not a reported evaluation result. Run Model Studio for a held-out benchmark.")
    for note in notes:
        print(f"DATA NOTE: {note}")


if __name__ == "__main__":
    main()
