"""
Iterative LLM taxonomy merge on top of embedding + large-K K-means buckets.

Uses the same SBERT model as Step 4 for embeddings, runs FAISS K-means with a
large K (default 1024), then processes micro-clusters in random order. Within
each cluster, rows are shuffled and fed in batches of size min(k, remaining).

``--merge_mode patch`` (default): the prompt sends a **bounded text outline** of
the existing tree and the LLM returns only ``create_nodes`` + ``place_rows``;
the script applies patches server-side (scales to large taxonomies).
``--merge_mode full_tree``: legacy behavior (full merged JSON each step).

Does not modify Step 4–6 scripts. Writes a parallel artifact layout under
``--output_dir`` so ``scripts/evaluate_pipeline.py`` can be pointed at the same
directory as ``--experiment_dir`` (expects ``models/all_extracted_discourse_with_clusters.csv``
and ``hierarchy_results__llm_iterative/``).

Example:
  python src/run_llm_iterative_hierarchy.py \\
    --input_data_file experiments/bbc_news/...json \\
    --trained_sbert_model_name experiments/bbc_news/models/.../trained-model \\
    --output_dir experiments/bbc_news_llm_hier \\
    --input_col_name label \\
    --model gpt-4o-mini \\
    --use_openai \\
    --experiment bbc-news
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from step_4__merge_labels import (  # noqa: E402
    assign_kmeans_clusters,
    read_dataframe,
    train_kmeans_clustering,
)
from utils_openai_client import load_model  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def resolve_row_id_column(df: pd.DataFrame, id_col: str | None) -> tuple[pd.DataFrame, str]:
    df = df.copy().reset_index(drop=True)
    if id_col is not None and id_col in df.columns:
        return df, id_col
    for cand in ("custom_id", "index"):
        if cand in df.columns:
            return df, cand
    df["__row_uid"] = df.index.astype(str)
    return df, "__row_uid"


def collect_leaf_row_ids(node: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(node, dict):
        return out
    lids = node.get("leaf_row_ids")
    if lids is not None:
        for x in lids:
            out.append(str(x))
    for ch in node.get("children") or []:
        out.extend(collect_leaf_row_ids(ch))
    return out


def _json_loads_dict(s: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def find_node_by_id(root: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    if root.get("id") == node_id:
        return root
    for ch in root.get("children") or []:
        found = find_node_by_id(ch, node_id)
        if found is not None:
            return found
    return None


def remove_row_ids_from_tree(node: dict[str, Any], to_remove: set[str]) -> None:
    lids = node.get("leaf_row_ids")
    if lids is not None:
        node["leaf_row_ids"] = [x for x in lids if str(x) not in to_remove]
    for ch in node.get("children") or []:
        remove_row_ids_from_tree(ch, to_remove)


def hierarchy_outline_for_prompt(
    node: dict[str, Any],
    max_chars: int = 14_000,
) -> str:
    """Bounded text outline so prompts stay small as the tree grows."""
    lines: list[str] = []

    def walk(n: dict[str, Any], indent: str) -> None:
        if sum(len(x) + 1 for x in lines) >= max_chars:
            return
        nid = str(n.get("id", "?"))
        name = str(n.get("name", ""))[:72]
        lids = n.get("leaf_row_ids")
        if lids is not None:
            lines.append(f"{indent}- {nid} ({name}) [leaf_bucket, {len(lids)} rows]")
        else:
            nc = len(n.get("children") or [])
            lines.append(f"{indent}- {nid} ({name}) [internal, {nc} children]")
        for ch in n.get("children") or []:
            walk(ch, indent + "  ")

    walk(node, "")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 40] + "\n... [outline truncated] ..."
    return out


def apply_patch_to_hierarchy(
    root: dict[str, Any],
    patch: dict[str, Any],
    batch_row_ids: set[str],
) -> None:
    """
    Mutates ``root`` in place: create nodes, then move batch_row_ids into buckets.
    """
    creates = patch.get("create_nodes") or []
    places = patch.get("place_rows") or []
    if not isinstance(creates, list) or not isinstance(places, list):
        raise ValueError("patch must have create_nodes and place_rows arrays")

    remaining = [c for c in creates if isinstance(c, dict)]
    max_iters = max(len(remaining) + 5, 12)
    for _ in range(max_iters):
        if not remaining:
            break
        next_remaining: list[dict[str, Any]] = []
        added = 0
        for spec in remaining:
            cid = str(spec.get("id", ""))
            pid = str(spec.get("parent_id", ""))
            if not cid or not pid:
                next_remaining.append(spec)
                continue
            if find_node_by_id(root, cid) is not None:
                continue
            parent = find_node_by_id(root, pid)
            if parent is None:
                next_remaining.append(spec)
                continue
            parent.setdefault("children", [])
            kind = spec.get("kind", "internal")
            name = str(spec.get("name", cid))[:200]
            if kind == "leaf_bucket":
                child = {"id": cid, "name": name, "leaf_row_ids": []}
            else:
                child = {"id": cid, "name": name, "children": []}
            parent["children"].append(child)
            added += 1
        remaining = next_remaining
        if not remaining:
            break
        if added == 0:
            raise ValueError(f"Unresolved create_nodes (missing parent or cycle?): {remaining[:5]!r}")
    if remaining:
        raise ValueError(f"Unresolved create_nodes after iterations: {remaining[:5]!r}")

    remove_row_ids_from_tree(root, batch_row_ids)

    seen_place: set[str] = set()
    for pr in places:
        if not isinstance(pr, dict):
            continue
        rid = str(pr.get("row_id", ""))
        bid = str(pr.get("leaf_bucket_id", ""))
        if not rid or not bid:
            continue
        if rid in seen_place:
            raise ValueError(f"duplicate row_id in place_rows: {rid}")
        if rid not in batch_row_ids:
            raise ValueError(f"place_rows row_id {rid} not in current batch")
        bucket = find_node_by_id(root, bid)
        if bucket is None:
            raise ValueError(f"leaf_bucket_id {bid} not found")
        if bucket.get("leaf_row_ids") is None:
            bucket["leaf_row_ids"] = []
        bucket["leaf_row_ids"].append(rid)
        seen_place.add(rid)
    if seen_place != batch_row_ids:
        missing = batch_row_ids - seen_place
        extra = seen_place - batch_row_ids
        raise ValueError(f"place_rows must cover batch ids exactly; missing={missing!r} extra={extra!r}")


def extract_json_tree(text: str) -> dict[str, Any] | None:
    """Parse first balanced {...} JSON object from model output (no noisy fallbacks)."""
    if not text or not text.strip():
        return None

    candidates: list[str] = []
    # Fenced ``` blocks (each may contain the tree)
    for block in re.findall(r"```(?:json)?\s*([\s\S]*?)```", text.strip()):
        b = block.strip()
        if b.startswith("{") or "{" in b:
            candidates.append(b)
    candidates.append(text.strip())

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        start = cand.find("{")
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(cand)):
            c = cand[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = cand[start : i + 1]
                    obj = _json_loads_dict(chunk)
                    if obj is not None:
                        return obj
                    break
        if depth != 0:
            logging.debug("extract_json_tree: unbalanced braces (truncated JSON?).")
    return None


def walk_leaf_paths(
    node: dict[str, Any], prefix_ids: list[str]
) -> list[tuple[str, list[str]]]:
    """Return (row_id, path_node_ids) where path includes root ... leaf bucket."""
    results: list[tuple[str, list[str]]] = []
    if not isinstance(node, dict):
        return results
    nid = str(node.get("id", ""))
    path_here = prefix_ids + [nid]
    lids = node.get("leaf_row_ids")
    if lids is not None:
        for rid in lids:
            results.append((str(rid), path_here))
    for ch in node.get("children") or []:
        results.extend(walk_leaf_paths(ch, path_here))
    return results


def export_eval_files(
    tree: dict[str, Any],
    df: pd.DataFrame,
    row_id_col: str,
    centroid_col: str,
    hierarchy_out: Path,
) -> None:
    """
    Write optimal_thresholds.csv and clusters_level_*_clusters.csv compatible with
    scripts/evaluate_pipeline.py (centroid_id -> fcluster_label via majority vote
    of row-level tree keys at each depth).
    """
    hierarchy_out.mkdir(parents=True, exist_ok=True)
    leaf_paths = walk_leaf_paths(tree, [])
    if not leaf_paths:
        logging.warning("No leaves in tree; skipping eval export.")
        return

    row_to_path = {rid: pth for rid, pth in leaf_paths}
    max_len = max(len(p) for _, p in leaf_paths)

    if max_len <= 1:
        level_indices = [1]
    else:
        level_indices = list(range(1, max_len))

    rows_meta = []
    for rid, path in row_to_path.items():
        sub = df.loc[df[row_id_col].astype(str) == str(rid)]
        if len(sub) == 0:
            logging.warning("Row id %s not in dataframe; skipping eval row.", rid)
            continue
        cid = int(sub.iloc[0][centroid_col])
        rows_meta.append((rid, cid, path))

    thresholds_rows = []
    for level in level_indices:
        keys_at_level = []
        vote_centroid: dict[int, list[str]] = {}
        for rid, cid, path in rows_meta:
            key = path[level] if level < len(path) else path[-1]
            keys_at_level.append(key)
            vote_centroid.setdefault(cid, []).append(key)

        uniq = sorted(set(keys_at_level))
        key_to_label = {k: i + 1 for i, k in enumerate(uniq)}
        k_clusters = len(uniq)

        assign_rows = []
        for cid in sorted(vote_centroid.keys()):
            votes = vote_centroid[cid]
            maj = Counter(votes).most_common(1)[0][0]
            assign_rows.append(
                {"centroid_id": cid, "fcluster_label": key_to_label[maj]}
            )

        pd.DataFrame(assign_rows).sort_values("centroid_id").to_csv(
            hierarchy_out / f"clusters_level_{level}_{k_clusters}_clusters.csv",
            index=False,
        )
        thresholds_rows.append(
            {
                "threshold": 1.0,
                "level": level,
                "n_clusters": k_clusters,
                "silhouette_score": 0.0,
            }
        )

    pd.DataFrame(thresholds_rows).to_csv(
        hierarchy_out / "optimal_thresholds.csv", index=False
    )
    logging.info(
        "Wrote %d hierarchy levels under %s",
        len(thresholds_rows),
        hierarchy_out,
    )


DEFAULT_RULES = """\
- Output MUST be exactly one JSON object (valid JSON). No markdown fences. No commentary after the JSON.
- Use COMPACT JSON: no indentation, minimal whitespace (the tree can get large).
- Root must use \"id\": \"root\" and include keys \"name\", \"id\", \"children\".
- Internal grouping nodes use \"name\", \"id\", \"children\".
- Leaf buckets use \"name\", \"id\", \"leaf_row_ids\": [ ... string ids ... ] (no empty leaf buckets).
- Every id in BATCH must appear exactly once across all leaf_row_ids in the tree.
- Every row id that already existed in the tree must still appear exactly once (you may move or regroup).
- Use short stable ids for internal nodes (e.g. \"n1\", \"n2\").
"""


def build_prompt(
    hierarchy: dict[str, Any],
    batch_items: list[dict[str, Any]],
    rules: str,
    retry_hint: str = "",
) -> str:
    batch_lines = []
    for it in batch_items:
        batch_lines.append(json.dumps(it, ensure_ascii=False, separators=(",", ":")))
    hier_compact = json.dumps(hierarchy, ensure_ascii=False, separators=(",", ":"))
    hint_block = f"\n{retry_hint}\n" if retry_hint else ""
    return f"""You merge datapoints into a hierarchical taxonomy.

RULES:
{rules.strip()}
{hint_block}
CURRENT HIERARCHY (JSON, compact):
{hier_compact}

NEW BATCH (each line is one JSON object with row_id, label, text):
{chr(10).join(batch_lines)}

Respond with the FULL merged hierarchy as ONE compact JSON object only (root id \"root\"). No markdown.
"""


DEFAULT_PATCH_RULES = """\
- Output ONE compact JSON object with keys \"create_nodes\" (array) and \"place_rows\" (array). No markdown or prose after the JSON.
- Root node id is \"root\" (already exists). Reference it as parent_id when attaching under the top level.
- create_nodes: {\"id\", \"name\", \"parent_id\", \"kind\"} where kind is \"internal\" or \"leaf_bucket\".
  Create only nodes you need for this batch; use short ids (n1, lb1). Parents must exist or appear earlier in create_nodes (parent before child).
- place_rows: {\"row_id\", \"leaf_bucket_id\"} — include EVERY row_id from THIS BATCH exactly once. leaf_bucket_id must be a node with kind leaf_bucket (create it if needed).
- Do NOT output the full tree; only patches for integrating this batch.
"""


def build_prompt_patch(
    hierarchy_root: dict[str, Any],
    batch_items: list[dict[str, Any]],
    rules: str,
    outline_max_chars: int,
    retry_hint: str = "",
) -> str:
    batch_lines = []
    for it in batch_items:
        batch_lines.append(json.dumps(it, ensure_ascii=False, separators=(",", ":")))
    outline = hierarchy_outline_for_prompt(hierarchy_root, max_chars=outline_max_chars)
    hint_block = f"\n{retry_hint}\n" if retry_hint else ""
    batch_ids = [str(it["row_id"]) for it in batch_items]
    return f"""You integrate new datapoints into an existing taxonomy using a small PATCH (not the full tree).

RULES:
{rules.strip()}
{hint_block}
HIERARCHY OUTLINE (existing structure; row counts only — full tree is not shown):
{outline}

THIS BATCH row_ids (each must appear exactly once in place_rows): {json.dumps(batch_ids)}

NEW BATCH (one JSON object per line with row_id, label, text):
{chr(10).join(batch_lines)}

Respond with ONLY the patch JSON object (create_nodes + place_rows).
"""


def call_llm_openai(prompt: str, model_name: str, temperature: float, max_tokens: int) -> str:
    _, client = load_model(model_name)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def call_llm_vllm(prompt: str, model_name: str, temperature: float, max_tokens: int) -> str:
    from utils_vllm_client import load_model as load_vllm_model
    from vllm import SamplingParams

    _tok, model = load_vllm_model(model_name)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = model.generate([prompt], sp)
    return outputs[0].outputs[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM iterative hierarchy on embedding + large-K K-means.")
    parser.add_argument("--input_data_file", type=str, required=True)
    parser.add_argument("--trained_sbert_model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_col_name", type=str, required=True, help="Column embedded with SBERT (same as Step 4).")
    parser.add_argument("--label_col", type=str, default="label", help="Semantic label shown to the LLM.")
    parser.add_argument("--text_col", type=str, default=None, help="Extra text for prompts (e.g. description). Optional.")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--ncentroids", type=int, default=1024)
    parser.add_argument("--kmeans_niter", type=int, default=200)
    parser.add_argument("--k_batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sbert_batch_size", type=int, default=100)
    parser.add_argument("--n_rows_to_process", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI API; default is vLLM.")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Max new tokens per LLM completion; raise if JSON is truncated.",
    )
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--rules", type=str, default=None, help="Extra rules appended to the default rule block.")
    parser.add_argument("--kmeans_downsample_to", type=int, default=None)
    parser.add_argument(
        "--merge_mode",
        type=str,
        choices=("patch", "full_tree"),
        default="patch",
        help="patch: LLM returns small create_nodes+place_rows (scales). full_tree: legacy full merged JSON.",
    )
    parser.add_argument(
        "--outline_max_chars",
        type=int,
        default=14_000,
        help="Max character budget for hierarchy outline in patch mode.",
    )
    args = parser.parse_args()

    cfg_path = HERE.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
            os.environ.setdefault("HF_TOKEN", cfg.get("HF_TOKEN", ""))

    out_dir = Path(args.output_dir)
    models_dir = out_dir / "models"
    hier_dir = out_dir / "hierarchy_results__llm_iterative"
    ckpt_dir = out_dir / "checkpoints_llm_hierarchy"
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rules_base = DEFAULT_PATCH_RULES if args.merge_mode == "patch" else DEFAULT_RULES
    rules = rules_base
    if args.rules:
        rules += "\n" + args.rules

    df = read_dataframe(args.input_data_file, args.experiment)
    df = df.dropna(subset=[args.input_col_name])
    if args.n_rows_to_process is not None:
        df = df.head(args.n_rows_to_process)

    df, row_id_col = resolve_row_id_column(df, args.id_col)
    logging.info(
        "Using row id column: %s (%d rows), merge_mode=%s",
        row_id_col,
        len(df),
        args.merge_mode,
    )

    rng = random.Random(args.seed)

    logging.info("Loading SBERT: %s", args.trained_sbert_model_name)
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(args.trained_sbert_model_name)
    embeddings = sbert.encode(
        df[args.input_col_name].astype(str).tolist(),
        show_progress_bar=True,
        batch_size=args.sbert_batch_size,
    )

    n_samples = len(df)
    ncentroids = min(args.ncentroids, max(n_samples, 1))
    if ncentroids < args.ncentroids:
        logging.warning(
            "Reduced ncentroids from %d to %d (dataset size).",
            args.ncentroids,
            ncentroids,
        )

    centroids_path = models_dir / "cluster_centroids.npy"
    logging.info("K-means (ncentroids=%d)...", ncentroids)
    kmeans = train_kmeans_clustering(
        embeddings,
        ncentroids=ncentroids,
        niter=args.kmeans_niter,
        save_path=str(centroids_path),
        downsample_to=args.kmeans_downsample_to,
    )
    clusters = assign_kmeans_clusters(embeddings, kmeans=kmeans)
    df["cluster"] = clusters.astype(int)

    step4_path = models_dir / "all_extracted_discourse_with_clusters.csv"
    df.to_csv(step4_path, index=False)
    logging.info("Wrote Step-4-style assignments to %s", step4_path)

    # Free GPU before vLLM: SBERT + batch encoding can leave most of VRAM in use.
    del sbert
    del embeddings
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    cluster_ids = sorted(df["cluster"].unique().tolist())
    rng.shuffle(cluster_ids)

    hierarchy: dict[str, Any] = {
        "name": "All",
        "id": "root",
        "children": [],
    }

    step_idx = 0
    for cid in tqdm(cluster_ids, desc="micro-clusters"):
        idxs = df.index[df["cluster"] == cid].tolist()
        rng.shuffle(idxs)
        pos = 0
        while pos < len(idxs):
            take = min(args.k_batch, len(idxs) - pos)
            batch_idx = idxs[pos : pos + take]
            pos += take

            prev_ids = set(collect_leaf_row_ids(hierarchy))
            batch_items = []
            new_ids: set[str] = set()
            for i in batch_idx:
                row = df.loc[i]
                rid = str(row[row_id_col])
                new_ids.add(rid)
                entry: dict[str, Any] = {
                    "row_id": rid,
                    "label": row[args.label_col] if args.label_col in df.columns else "",
                    "text": str(row[args.input_col_name]),
                }
                if args.text_col and args.text_col in df.columns:
                    entry["extra"] = str(row[args.text_col])
                batch_items.append(entry)

            expected = prev_ids | new_ids
            merged: dict[str, Any] | None = None
            last_raw = ""

            if args.merge_mode == "patch":
                batch_row_ids = set(new_ids)
                for attempt in range(args.max_retries):
                    retry_hint = ""
                    if attempt > 0:
                        retry_hint = (
                            "PREVIOUS ATTEMPT FAILED: Output valid JSON only with keys "
                            "create_nodes and place_rows. Cover every batch row_id exactly once."
                        )
                    prompt = build_prompt_patch(
                        hierarchy,
                        batch_items,
                        rules,
                        args.outline_max_chars,
                        retry_hint=retry_hint,
                    )
                    if args.use_openai:
                        raw = call_llm_openai(
                            prompt, args.model_name, args.temperature, args.max_tokens
                        )
                    else:
                        raw = call_llm_vllm(
                            prompt, args.model_name, args.temperature, args.max_tokens
                        )
                    last_raw = raw
                    patch = extract_json_tree(raw)
                    if patch is None:
                        logging.warning("Attempt %d: failed to parse patch JSON.", attempt + 1)
                        tail = raw[-400:] if raw else ""
                        logging.warning("Last 400 chars of model output: %r", tail)
                        continue
                    if not isinstance(patch.get("create_nodes"), list) or not isinstance(
                        patch.get("place_rows"), list
                    ):
                        logging.warning("Attempt %d: patch missing create_nodes or place_rows lists.", attempt + 1)
                        continue
                    trial = copy.deepcopy(hierarchy)
                    try:
                        apply_patch_to_hierarchy(trial, patch, batch_row_ids)
                    except ValueError as e:
                        logging.warning("Attempt %d: invalid patch: %s", attempt + 1, e)
                        continue
                    got = set(collect_leaf_row_ids(trial))
                    if got != expected:
                        logging.warning(
                            "Attempt %d: leaf id mismatch after patch (got %d, expected %d).",
                            attempt + 1,
                            len(got),
                            len(expected),
                        )
                        continue
                    hierarchy = trial
                    merged = patch
                    break

                if merged is None:
                    fail_path = ckpt_dir / f"failed_raw_step_{step_idx:06d}_cluster_{cid}.txt"
                    try:
                        fail_path.write_text(last_raw or "", encoding="utf-8")
                        logging.error("Wrote failing model output to %s", fail_path)
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"Failed patch merge for cluster {cid}, step {step_idx}. "
                        "See failed_raw_step_* and increase --max_retries or check model output."
                    )

            else:
                for attempt in range(args.max_retries):
                    retry_hint = ""
                    if attempt > 0:
                        retry_hint = (
                            "PREVIOUS ATTEMPT FAILED: Output must be valid compact JSON only. "
                            "Do not truncate; shorten labels if needed."
                        )
                    prompt = build_prompt(hierarchy, batch_items, rules, retry_hint=retry_hint)
                    if args.use_openai:
                        raw = call_llm_openai(
                            prompt, args.model_name, args.temperature, args.max_tokens
                        )
                    else:
                        raw = call_llm_vllm(
                            prompt, args.model_name, args.temperature, args.max_tokens
                        )
                    last_raw = raw
                    merged = extract_json_tree(raw)
                    if merged is None:
                        logging.warning("Attempt %d: failed to parse JSON.", attempt + 1)
                        tail = raw[-400:] if raw else ""
                        logging.warning("Last 400 chars of model output: %r", tail)
                        continue
                    got = set(collect_leaf_row_ids(merged))
                    if got != expected:
                        logging.warning(
                            "Attempt %d: leaf id mismatch (got %d, expected %d).",
                            attempt + 1,
                            len(got),
                            len(expected),
                        )
                        continue
                    root_id = merged.get("id")
                    if root_id != "root":
                        logging.warning("Attempt %d: root id was %r, fixing to root.", attempt + 1, root_id)
                        merged["id"] = "root"
                    hierarchy = merged
                    break

                if merged is None or set(collect_leaf_row_ids(hierarchy)) != expected:
                    fail_path = ckpt_dir / f"failed_raw_step_{step_idx:06d}_cluster_{cid}.txt"
                    try:
                        fail_path.write_text(last_raw or "", encoding="utf-8")
                        logging.error("Wrote failing model output to %s", fail_path)
                    except OSError:
                        pass
                    raise RuntimeError(
                        f"Failed to merge batch for cluster {cid}, step {step_idx}. "
                        "Increase --max_tokens; ensure VLLM_MAX_MODEL_LEN fits prompt+output "
                        "(e.g. 32768). See failed_raw_step_* in checkpoints."
                    )

            step_idx += 1
            ck_path = ckpt_dir / f"step_{step_idx:06d}.json"
            with open(ck_path, "w", encoding="utf-8") as f:
                json.dump(hierarchy, f, indent=2, ensure_ascii=False)

    final_tree_path = out_dir / "hierarchy_llm_final.json"
    with open(final_tree_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    logging.info("Wrote final hierarchy to %s", final_tree_path)

    export_eval_files(hierarchy, df, row_id_col, "cluster", hier_dir)

    manifest = {
        "input_data_file": str(args.input_data_file),
        "trained_sbert_model_name": args.trained_sbert_model_name,
        "ncentroids": ncentroids,
        "k_batch": args.k_batch,
        "seed": args.seed,
        "model_name": args.model_name,
        "use_openai": args.use_openai,
        "n_rows": len(df),
        "row_id_col": row_id_col,
        "final_tree": str(final_tree_path),
        "hierarchy_eval_dir": str(hier_dir),
        "step4_csv": str(step4_path),
        "merge_mode": args.merge_mode,
        "outline_max_chars": args.outline_max_chars,
    }
    with open(out_dir / "manifest_llm_hierarchy.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Done. Point evaluate_pipeline.py --experiment_dir to %s", out_dir)


if __name__ == "__main__":
    main()
