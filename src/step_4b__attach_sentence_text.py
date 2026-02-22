#!/usr/bin/env python3
"""
Attach raw sentence text to the clustered discourse file.

This script joins `all_extracted_discourse_with_clusters.csv` (from Step 4)
with the raw input sentences (e.g., processed_custom_input.csv) by parsing
the `index` field (doc_index + sent_index) and merging on those keys.

Typical usage:
python src/step_4b__attach_sentence_text.py \
  --clustered_file experiments/wiki_biographies_10000/models/all_extracted_discourse_with_clusters.csv \
  --raw_sentences_file processed_custom_input.csv \
  --output_file experiments/wiki_biographies_10000/models/all_extracted_discourse_with_clusters_and_text.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Tuple

import pandas as pd


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


INDEX_RE = re.compile(r"^(?P<doc>.+?)__sent_(?:index|idx)-(?P<sent>\d+)$")


def parse_index(index_val: str) -> Tuple[str | None, int | None]:
    if not isinstance(index_val, str):
        return None, None
    m = INDEX_RE.match(index_val)
    if not m:
        return None, None
    doc = m.group("doc")
    sent = int(m.group("sent"))
    return doc, sent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Attach raw sentence_text to clustered discourse CSV using doc_index + sent_index parsed from index."
    )
    ap.add_argument(
        "--clustered_file",
        required=True,
        help="Path to clustered discourse CSV (Step 4 output).",
    )
    ap.add_argument(
        "--raw_sentences_file",
        required=True,
        help="Path to raw sentences CSV (e.g., processed_custom_input.csv).",
    )
    ap.add_argument(
        "--output_file",
        required=True,
        help="Path to write output CSV.",
    )
    ap.add_argument(
        "--index_col",
        default="index",
        help="Column name in clustered_file containing doc/sent index (default: index).",
    )
    ap.add_argument(
        "--raw_doc_col",
        default="doc_index",
        help="Column name in raw_sentences_file for document id (default: doc_index).",
    )
    ap.add_argument(
        "--raw_sent_col",
        default="sent_index",
        help="Column name in raw_sentences_file for sentence index (default: sent_index).",
    )
    ap.add_argument(
        "--raw_text_col",
        default="sentence_text",
        help="Column name in raw_sentences_file with raw sentence text (default: sentence_text).",
    )
    ap.add_argument(
        "--out_text_col",
        default="sentence_text",
        help="Column name to create in output (default: sentence_text).",
    )
    ap.add_argument(
        "--backup_suffix",
        default=".bak",
        help="If output_file == clustered_file, create a backup with this suffix (default: .bak).",
    )
    args = ap.parse_args()

    clustered_path = args.clustered_file
    raw_path = args.raw_sentences_file
    out_path = args.output_file

    logging.info("Loading clustered file: %s", clustered_path)
    clustered_df = pd.read_csv(clustered_path)
    if args.index_col not in clustered_df.columns:
        raise KeyError(f"Column '{args.index_col}' not found in {clustered_path}.")

    logging.info("Loading raw sentences file: %s", raw_path)
    raw_df = pd.read_csv(raw_path)
    for col in [args.raw_doc_col, args.raw_sent_col, args.raw_text_col]:
        if col not in raw_df.columns:
            raise KeyError(f"Column '{col}' not found in {raw_path}.")

    # Parse doc_index and sent_index from clustered index
    logging.info("Parsing doc_index + sent_index from clustered index...")
    parsed = clustered_df[args.index_col].apply(parse_index)
    clustered_df["_doc_index"] = parsed.apply(lambda x: x[0])
    clustered_df["_sent_index"] = parsed.apply(lambda x: x[1])

    missing_parsed = clustered_df["_doc_index"].isna().sum()
    if missing_parsed > 0:
        logging.warning(
            "Failed to parse %s rows from '%s'. They will have NaN sentence_text.",
            missing_parsed,
            args.index_col,
        )

    # Merge raw sentences
    logging.info("Merging raw sentence text...")
    raw_subset = raw_df[[args.raw_doc_col, args.raw_sent_col, args.raw_text_col]].copy()
    raw_subset = raw_subset.rename(
        columns={
            args.raw_doc_col: "_doc_index",
            args.raw_sent_col: "_sent_index",
            args.raw_text_col: args.out_text_col,
        }
    )

    merged = clustered_df.merge(
        raw_subset,
        on=["_doc_index", "_sent_index"],
        how="left",
        validate="m:1",
    )

    missing_text = merged[args.out_text_col].isna().sum()
    logging.info(
        "Attached sentence_text. Missing raw text for %s / %s rows.",
        missing_text,
        len(merged),
    )

    # Drop helper columns
    merged = merged.drop(columns=["_doc_index", "_sent_index"])

    # Backup if overwriting
    if os.path.abspath(out_path) == os.path.abspath(clustered_path):
        backup_path = clustered_path + args.backup_suffix
        logging.info("Overwriting clustered_file. Creating backup at: %s", backup_path)
        if not os.path.exists(backup_path):
            os.rename(clustered_path, backup_path)
        else:
            logging.warning("Backup file already exists: %s (not overwriting).", backup_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged.to_csv(out_path, index=False)
    logging.info("Wrote output: %s", out_path)


if __name__ == "__main__":
    main()
