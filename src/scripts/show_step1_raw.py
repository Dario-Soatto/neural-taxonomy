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
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched: {args.pattern}")

    shown = 0
    for path in files:
        for row in iter_rows(path):
            raw = row.get("response_raw")
            if raw:
                print(f"{path} | index={row.get('index')} | raw={raw!r}")
                shown += 1
                if shown >= args.n:
                    return

    print("No response_raw found.")


if __name__ == "__main__":
    main()
