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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--sample-nonnull", type=int, default=0)
    args = ap.parse_args()

    total = 0
    null_resp = 0
    nonnull = 0
    for row in iter_rows(args.path):
        total += 1
        resp = row.get("response", None)
        if resp is None:
            null_resp += 1
            continue
        nonnull += 1
        if args.sample_nonnull and nonnull <= args.sample_nonnull:
            norm = normalize_response(row)
            print(f"NONNULL[{nonnull}]: {norm}")

    print("total", total, "response_null", null_resp, "response_nonnull", nonnull)


if __name__ == "__main__":
    main()
