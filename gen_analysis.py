"""
Quick-and-dirty analyser for RAG chatbot interaction logs.

The input JSON is the *array* structure you showed, e.g.
[
  { "query_id": "...", "user_query": "...", ... },
  { ... },
  ...
]
"""

import json
from pathlib import Path
from statistics import mean, median, quantiles
from collections import Counter, defaultdict
import argparse
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# 1. Helpers
# ────────────────────────────────────────────────────────────────────
def load_interactions(json_path: Path) -> list[dict]:
    """Read and return the list of interaction dicts."""
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def latency_stats(latencies: list[int]) -> dict[str, float]:
    """Return common latency percentiles (ms)."""
    q = quantiles(latencies, n=100)      # percentiles 1…99
    return {
        "count": len(latencies),
        "mean":  mean(latencies),
        "median": median(latencies),
        "p90":   q[89],
        "p95":   q[94],
        "p99":   q[98],
        "max":   max(latencies),
    }


def feedback_summary(feedback: list[str]) -> dict[str, int]:
    """Count thumb_up / thumb_down / etc."""
    return Counter(feedback)


def source_usage(records: list[dict]) -> pd.DataFrame:
    """
    Flatten retrieved_chunks and produce a DataFrame with:
      query_id | rank | source | retrieval_score | user_feedback
    """
    rows = []
    for r in records:
        for rank, ch in enumerate(r["retrieved_chunks"], 1):   # rank 1..k
            rows.append(
                {
                    "query_id": r["query_id"],
                    "rank": rank,
                    "source": ch["source"],
                    "retrieval_score": ch["retrieval_score"],
                    "user_feedback": r.get("user_feedback", ""),
                }
            )
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# 2. Main analysis routine
# ────────────────────────────────────────────────────────────────────
def analyse(json_file: Path) -> None:
    interactions = load_interactions(json_file)

    # 2-A ─ Latency
    lat_ms = [r["response_latency_ms"] for r in interactions]
    lat_stat = latency_stats(lat_ms)

    # 2-B ─ User feedback
    fb_stat = feedback_summary([r.get("user_feedback", "") for r in interactions])

    # 2-C ─ Retrieval inspection
    df_retr = source_usage(interactions)

    #   · source share overall
    share_overall = df_retr["source"].value_counts(normalize=True) * 100

    #   · top-rank (rank 1) source share
    share_top1 = (
        df_retr[df_retr["rank"] == 1]["source"].value_counts(normalize=True) * 100
    )

    #   · mean retrieval score per source
    mean_score = df_retr.groupby("source")["retrieval_score"].mean().sort_values(
        ascending=False
    )

    #   · feedback vs. sources (how often thumbs-down when top chunk
    #     comes from Source 3, etc.)
    fb_vs_source = (
        df_retr[df_retr["rank"] == 1]
        .groupby(["source", "user_feedback"])
        .size()
        .unstack(fill_value=0)
    )

    # ──────────────────── Pretty-print ────────────────────
    print("\n=== Latency (ms) ===")
    for k, v in lat_stat.items():
        print(f"{k:>8}: {v:,.2f}")

    print("\n=== User feedback counts ===")
    for k, v in fb_stat.items():
        print(f"{k:>12}: {v}")

    print("\n=== Source share across *all* retrieved chunks ===")
    print(share_overall.round(2).astype(str) + " %")

    print("\n=== Source share when the chunk is RANK 1 ===")
    print(share_top1.round(2).astype(str) + " %")

    print("\n=== Mean retrieval_score by source ===")
    print(mean_score.round(3))

    print("\n=== Thumb-down vs. top-rank source (pivot) ===")
    print(fb_vs_source)


# ────────────────────────────────────────────────────────────────────
# 3. CLI entry-point
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analyse RAG interaction logs")
    ap.add_argument("json_file", type=Path, help="Path to interactions JSON")
    args = ap.parse_args()

    analyse(args.json_file)
