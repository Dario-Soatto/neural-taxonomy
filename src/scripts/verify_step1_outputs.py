import argparse
import glob
import json


def iter_rows(path):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True, help="Glob for Step 1 shard files")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern}")

    total = 0
    response_null = 0
    response_nonnull = 0
    for path in files:
        for row in iter_rows(path):
            total += 1
            resp = row.get("response", None)
            if resp is None:
                response_null += 1
            else:
                response_nonnull += 1

    print("files", len(files))
    print("rows", total)
    print("response_null", response_null)
    print("response_nonnull", response_nonnull)


if __name__ == "__main__":
    main()
