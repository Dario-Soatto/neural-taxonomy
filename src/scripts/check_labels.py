import argparse
import json


def iter_rows(path):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def normalize_response(row):
    resp = row.get("response", row)
    if resp is None:
        return {}
    if isinstance(resp, list):
        resp = resp[0] if resp else {}
    if resp is None:
        return {}
    return resp


def sample_rows(path, n):
    for i, row in zip(range(n), iter_rows(path)):
        resp = normalize_response(row)
        label = resp.get("label")
        desc = resp.get("description")
        sent = row.get("sent_text") or row.get("sentence_text")
        print(f"{i:02d} | label={label!r} | desc={desc!r} | sent={sent!r}")


def count_missing(path):
    total = 0
    missing = 0
    for row in iter_rows(path):
        total += 1
        resp = normalize_response(row)
        label = (resp.get("label") or "").strip()
        desc = (resp.get("description") or "").strip()
        if not label or not desc:
            missing += 1
    print("total", total, "missing_label_or_desc", missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--count-missing", action="store_true")
    ap.add_argument("--raw", type=int, default=0)
    ap.add_argument("--show-keys", action="store_true")
    args = ap.parse_args()

    if args.sample:
        sample_rows(args.path, args.sample)
    if args.count_missing:
        count_missing(args.path)
    if args.raw:
        for i, row in zip(range(args.raw), iter_rows(args.path)):
            print(f"RAW[{i}]: {row}")
    if args.show_keys:
        for row in iter_rows(args.path):
            print("keys:", sorted(row.keys()))
            resp = row.get("response", row)
            if isinstance(resp, dict):
                print("response keys:", sorted(resp.keys()))
            else:
                print("response type:", type(resp).__name__)
            break


if __name__ == "__main__":
    main()
