"""
Prepares the BBC News dataset for the 6-step pipeline.

Downloads SetFit/bbc-news from HuggingFace (2,225 articles, 5 categories),
concatenates train+test splits, and writes a CSV with columns:
  id, text, true_label

true_label is kept for post-hoc evaluation but is NOT used by the pipeline.

Usage:
  python src/scripts/prepare_bbc_news_data.py \
      --output_csv experiments/bbc_news/data/bbc_news.csv
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to write the output CSV.")
    args = parser.parse_args()

    from datasets import load_dataset, concatenate_datasets
    import pandas as pd

    logging.info("Loading SetFit/bbc-news from HuggingFace...")
    ds = load_dataset("SetFit/bbc-news")

    combined = concatenate_datasets([ds["train"], ds["test"]])
    logging.info("Combined train+test: %d documents.", len(combined))

    df = pd.DataFrame({
        "id": range(len(combined)),
        "text": combined["text"],
        "true_label": combined["label_text"],
    })

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logging.info("Wrote %d rows to %s.", len(df), out_path)
    logging.info("Label distribution:\n%s", df["true_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
