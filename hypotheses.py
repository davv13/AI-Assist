#!/usr/bin/env python3
"""
Validation-suite for five RAG hypotheses

H-1  Prompt-bloat latency
H-2  Source-3 top-rank → thumbs-down
H-3  High retrieval score but still thumbs-down
H-4  Low retrieval score drives thumbs-down
H-5  Long waits make users less forgiving
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from tabulate import tabulate

# --------------------------------------------------------------------------- #
# 1 · Source-name canonicalisation
# --------------------------------------------------------------------------- #
SOURCE_ALIAS = {
    "engineering wiki":            "Source 1",
    "confluence":                  "Source 2",
    "archived design docs (pdfs)": "Source 3",
}


def canonise_source(col: pd.Series) -> pd.Series:
    return (
        col.str.lower()
           .str.strip()
           .map(SOURCE_ALIAS)
           .fillna(col)
    )


# --------------------------------------------------------------------------- #
# 2 · Log loader
# --------------------------------------------------------------------------- #
def load_logs(path: Path):
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    root = pd.json_normalize(raw, sep="_")

    chunks = (
        root[["query_id", "retrieved_chunks"]]
        .explode("retrieved_chunks")
        .dropna(subset=["retrieved_chunks"])
    )

    chunk_df = (
        pd.json_normalize(chunks["retrieved_chunks"])
        .join(chunks[["query_id"]])
    )

    if "source" in chunk_df.columns:
        chunk_df["source"] = canonise_source(chunk_df["source"])

    return root, chunk_df


# --------------------------------------------------------------------------- #
# 3 · Helper
# --------------------------------------------------------------------------- #
def safe_mwu(a: pd.Series, b: pd.Series) -> float:
    try:
        return stats.mannwhitneyu(a, b, alternative="greater").pvalue
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------- #
# 4 · Hypothesis tests
# --------------------------------------------------------------------------- #
# H-1  Prompt-bloat latency
def test_h1(root, chunk_df):
    root["has_src3"] = root["query_id"].isin(
        chunk_df.loc[chunk_df["source"] == "Source 3", "query_id"]
    )
    lat_src3 = root.loc[root["has_src3"],  "response_latency_ms"]
    lat_no3  = root.loc[~root["has_src3"], "response_latency_ms"]
    if lat_src3.empty or lat_no3.empty:
        return {"note": "No queries contained Source 3 chunks."}
    p = safe_mwu(lat_src3, lat_no3)
    return {
        "mean_latency_src3": lat_src3.mean(),
        "mean_latency_no3":  lat_no3.mean(),
        "p_value_mwu":       p,
        "n_src3":            len(lat_src3),
        "n_no3":             len(lat_no3),
    }


# H-2  Source-3 ranked first → thumbs-down
def test_h2(root, chunk_df):
    top_ranks = chunk_df.groupby("query_id", sort=False).nth(0).reset_index()
    joined = root[["query_id", "user_feedback"]].merge(
        top_ranks[["query_id", "source"]], on="query_id"
    )
    ctab = pd.crosstab(joined["source"], joined["user_feedback"])
    chi2, p, *_ = stats.chi2_contingency(ctab)
    return {"contingency": ctab, "chi2": chi2, "p_value": p}


# H-3  High retrieval score but still thumbs-down
def test_h3(root, chunk_df, score_thresh: float = 0.9):
    max_scores = (
        chunk_df.groupby("query_id")["retrieval_score"].max().reset_index()
    )
    merged = root.merge(max_scores, on="query_id", how="left")
    high_hits = merged.loc[merged["retrieval_score"] >= score_thresh]

    ctab = pd.crosstab(high_hits["user_feedback"], columns="count")
    thumb_down_rate = (
        ctab.loc["thumb_down", "count"] / len(high_hits)
        if "thumb_down" in ctab.index
        else 0.0
    )
    return {
        "thumb_down_rate_high_retrieval": thumb_down_rate,
        "n_high_retrieval":               len(high_hits),
        "feedback_breakdown":             ctab,
    }


# H-4  Low retrieval score → more thumbs-down
def test_h4(root, chunk_df, thresh: float = 0.85):
    max_scores = (
        chunk_df.groupby("query_id")["retrieval_score"].max().reset_index()
    )
    merged = root.merge(max_scores, on="query_id", how="left")
    low  = merged.loc[merged["retrieval_score"] <  thresh]
    high = merged.loc[merged["retrieval_score"] >= thresh]

    if low.empty or high.empty:
        return {"note": "One of the score buckets is empty."}

    ctab = pd.crosstab(
        merged["retrieval_score"] < thresh,
        merged["user_feedback"]
    )
    chi2, p, *_ = stats.chi2_contingency(ctab)

    low_td_rate  = (low["user_feedback"] == "thumb_down").mean()
    high_td_rate = (high["user_feedback"] == "thumb_down").mean()

    return {
        "thumb_down_rate_low":   low_td_rate,
        "thumb_down_rate_high":  high_td_rate,
        "p_value":               p,
        "n_low":                 len(low),
        "n_high":                len(high),
    }


# H-5  Long waits → more thumbs-down
def test_h5(root):
    bins = [0, 2000, 4000, float("inf")]
    labels = ["fast(<2s)", "mid(2-4s)", "slow(>4s)"]
    root["latency_bucket"] = pd.cut(
        root["response_latency_ms"], bins=bins, labels=labels, right=False
    )
    ctab = pd.crosstab(root["latency_bucket"], root["user_feedback"])
    chi2, p, *_ = stats.chi2_contingency(ctab)

    td_rates = (
        ctab["thumb_down"] / ctab.sum(axis=1)
    ).rename("thumb_down_rate").to_dict()

    return {"bucket_rates": td_rates, "chi2": chi2, "p_value": p}


# --------------------------------------------------------------------------- #
# 5 · Entry-point
# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print("Usage: python rag_validation_suite.py  <interactions.json>")
        sys.exit(1)

    root, chunks = load_logs(Path(sys.argv[1]))

    results = {
        "H1_prompt_bloat_latency":   test_h1(root, chunks),
        "H2_source3_toprank_neg":    test_h2(root, chunks),
        "H3_high_score_neg":         test_h3(root, chunks),
        "H4_low_score_neg":          test_h4(root, chunks),
        "H5_latency_vs_feedback":    test_h5(root),
    }

    print("\nVALIDATION SUMMARY\n" + "-" * 60)
    for name, res in results.items():
        print(f"\n{name}")
        for k, v in res.items():
            if isinstance(v, pd.DataFrame):
                print(tabulate(v, headers="keys", tablefmt="github"))
            else:
                print(f"{k:35}: {v}")
    print("-" * 60)


if __name__ == "__main__":
    main()
