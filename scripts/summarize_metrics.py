"""汇总 artifacts/metrics/*.pkl 到 CSV，便于论文表格。"""

from __future__ import annotations

import pickle
from pathlib import Path
import csv


def main() -> None:
    root = Path("artifacts/metrics")
    out_csv = Path("artifacts/tables/metrics_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(root.glob("*.pkl")):
        with p.open("rb") as f:
            d = pickle.load(f)
        row = {"exp_name": p.stem}
        row.update(d)
        rows.append(row)

    if not rows:
        print("no metrics found")
        return

    keys = [
        "exp_name",
        "test_accuracy",
        "test_macro_f1",
        "train_mean_active_ratio",
        "test_mean_active_ratio",
        "train_mean_encode_ms",
        "test_mean_encode_ms",
        "n_train",
        "n_test",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})

    print(f"saved summary: {out_csv}")


if __name__ == "__main__":
    main()
