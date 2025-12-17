#!/usr/bin/env python3
"""
Arrow-based LLM evaluation metrics for DuckDB.

Changes vs previous version:
- Each metric is implemented as an Arrow UDF that operates on Arrow batches.
- CSV batch mode removed; only DuckDB table updates are supported.
- Heavy models (embeddings, BLEURT, NLI) run batched over Arrow columns.
"""
import pandas as pd
from renumics import spotlight
import time
import os
import re
from collections import Counter
from typing import List, Tuple, Dict, Any
import time
import numpy as np
import duckdb
import pyarrow as pa
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# -------------------------------------------------------------------
# Global env / device settings
# -------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------
# Normalization & Tokenization (scalar helpers)
# -------------------------------------------------------------------

def normalize_text(text: str | None) -> str:
    """Lowercase, strip, collapse spaces."""
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str | None) -> List[str]:
    """Simple whitespace tokenization after normalization."""
    text = normalize_text(text)
    if not text:
        return []
    return text.split()


# -------------------------------------------------------------------
# Scalar metric helpers (as before)
# -------------------------------------------------------------------

def exact_match_scalar(gold: str | None, pred: str | None) -> float:
    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0


def prf1_token_overlap_scalar(gold: str | None, pred: str | None) -> Tuple[float, float, float]:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    if not gold_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0  # both empty -> perfect

    gold_counts = Counter(gold_tokens)
    pred_counts = Counter(pred_tokens)

    overlap = sum((gold_counts & pred_counts).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gold_tokens) if gold_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def rouge_n_scalar(gold: str | None, pred: str | None, n: int) -> Tuple[float, float, float]:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    gold_ngrams = ngrams(gold_tokens, n)
    pred_ngrams = ngrams(pred_tokens, n)

    if not gold_ngrams and not pred_ngrams:
        return 1.0, 1.0, 1.0

    gold_counts = Counter(gold_ngrams)
    pred_counts = Counter(pred_ngrams)

    overlap = sum((gold_counts & pred_counts).values())

    precision = overlap / len(pred_ngrams) if pred_ngrams else 0.0
    recall = overlap / len(gold_ngrams) if gold_ngrams else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def bleu_scalar(gold: str | None, pred: str | None) -> float:
    gold_tokens = tokenize(gold)
    pred_tokens = tokenize(pred)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not pred_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothing))


def levenshtein_scalar(a: str | None, b: str | None) -> int:
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = curr_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ca != cb)
            curr_row.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = curr_row
    return prev_row[-1]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


# -------------------------------------------------------------------
# Heavy models (global, reused across Arrow batches)
# -------------------------------------------------------------------

# Sentence embeddings
_EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# BLEURT
_BLEURT_MODEL_NAME = "Elron/bleurt-base-512"
_BLEURT_DEVICE = torch.device("cpu")
_BLEURT_TOKENIZER = AutoTokenizer.from_pretrained(_BLEURT_MODEL_NAME)
_BLEURT_MODEL = AutoModelForSequenceClassification.from_pretrained(
    _BLEURT_MODEL_NAME
).to(_BLEURT_DEVICE)
_BLEURT_MODEL.eval()

# NLI pipeline (BART MNLI)
_NLI_PIPE = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1,  # CPU
)


# -------------------------------------------------------------------
# Arrow helpers
# -------------------------------------------------------------------

def _arrow_to_pylist(arr: pa.Array | pa.ChunkedArray) -> List[str | None]:
    """
    Convert an Arrow Array or ChunkedArray to a Python list, preserving None for nulls.
    DuckDB Arrow UDFs often pass ChunkedArray, so we rely on to_pylist() which
    handles chunking and nulls correctly.
    """
    return arr.to_pylist()

# -------------------------------------------------------------------
# Arrow UDFs: lexical metrics
# Each UDF operates on whole Arrow column chunks at once.
# -------------------------------------------------------------------

def exact_match_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started exact match")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = [
        exact_match_scalar(g, p) for g, p in zip(gold, pred)
    ]
    return pa.array(out, type=pa.float64())


def token_precision_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token-pre")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        pr, _, _ = prf1_token_overlap_scalar(g, p)
        out.append(pr)
    return pa.array(out, type=pa.float64())


def token_recall_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token- rec")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, rc, _ = prf1_token_overlap_scalar(g, p)
        out.append(rc)
    return pa.array(out, type=pa.float64())


def token_f1_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token f1")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, _, f1 = prf1_token_overlap_scalar(g, p)
        out.append(f1)
    return pa.array(out, type=pa.float64())


def rouge1_precision_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge1")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        pr, _, _ = rouge_n_scalar(g, p, n=1)
        out.append(pr)
    return pa.array(out, type=pa.float64())


def rouge1_recall_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge1")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, rc, _ = rouge_n_scalar(g, p, n=1)
        out.append(rc)
    return pa.array(out, type=pa.float64())


def rouge1_f1_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge1")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, _, f1 = rouge_n_scalar(g, p, n=1)
        out.append(f1)
    return pa.array(out, type=pa.float64())


def rouge2_precision_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge2")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        pr, _, _ = rouge_n_scalar(g, p, n=2)
        out.append(pr)
    return pa.array(out, type=pa.float64())


def rouge2_recall_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge2")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, rc, _ = rouge_n_scalar(g, p, n=2)
        out.append(rc)
    return pa.array(out, type=pa.float64())


def rouge2_f1_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token rouge1")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = []
    for g, p in zip(gold, pred):
        _, _, f1 = rouge_n_scalar(g, p, n=2)
        out.append(f1)
    return pa.array(out, type=pa.float64())


def bleu_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started token bleu")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[float] = [bleu_scalar(g, p) for g, p in zip(gold, pred)]
    return pa.array(out, type=pa.float64())


def edit_distance_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("edit")
    gold = _arrow_to_pylist(gold_arr)
    pred = _arrow_to_pylist(pred_arr)
    out: List[int] = [levenshtein_scalar(g, p) for g, p in zip(gold, pred)]
    return pa.array(out, type=pa.int64())


# -------------------------------------------------------------------
# Arrow UDFs: embedding similarity, BLEURT, NLI (batched)
# -------------------------------------------------------------------

def embedding_similarity_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("embedding similarity")
    gold = [normalize_text(x) for x in _arrow_to_pylist(gold_arr)]
    pred = [normalize_text(x) for x in _arrow_to_pylist(pred_arr)]

    # Encode all sentences in one go for efficiency.
    # We concatenate then split back.
    all_texts = gold + pred
    embeddings = _EMB_MODEL.encode(all_texts, batch_size=32)
    emb_gold = embeddings[: len(gold)]
    emb_pred = embeddings[len(gold) :]

    out: List[float] = [
        cosine_similarity(np.array(g), np.array(p))
        for g, p in zip(emb_gold, emb_pred)
    ]
    return pa.array(out, type=pa.float64())


def bleurt_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started blue")
    gold = [normalize_text(x) for x in _arrow_to_pylist(gold_arr)]
    pred = [normalize_text(x) for x in _arrow_to_pylist(pred_arr)]

    # Handle empty pairs explicitly (BLEURT expects some content).
    pairs = [(g, p) for g, p in zip(gold, pred)]
    # Build tokenizer batch
    inputs = _BLEURT_TOKENIZER(
        [g for g, _ in pairs],
        [p for _, p in pairs],
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(_BLEURT_DEVICE)

    with torch.no_grad():
        outputs = _BLEURT_MODEL(**inputs)
        logits = outputs.logits.squeeze(-1).cpu().numpy()

    out = logits.astype(np.float64).tolist()
    return pa.array(out, type=pa.float64())


def nli_label_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started nli")
    gold = [normalize_text(x) for x in _arrow_to_pylist(gold_arr)]
    pred = [normalize_text(x) for x in _arrow_to_pylist(pred_arr)]

    # NLI pipeline supports batched inputs.
    inputs = [{"text": g, "text_pair": p} for g, p in zip(gold, pred)]
    results = _NLI_PIPE(inputs, truncation=True)

    labels: List[str] = []
    for res in results:
        if isinstance(res, list):
            # Some HF pipelines return list per example; take best
            best = max(res, key=lambda x: x.get("score", 0.0))
        else:
            best = res
        label = str(best.get("label", "UNKNOWN")).upper()
        labels.append(label)
    return pa.array(labels, type=pa.string())


def nli_label_score_arrow(gold_arr: pa.Array, pred_arr: pa.Array) -> pa.Array:
    print("started nli")
    gold = [normalize_text(x) for x in _arrow_to_pylist(gold_arr)]
    pred = [normalize_text(x) for x in _arrow_to_pylist(pred_arr)]

    inputs = [{"text": g, "text_pair": p} for g, p in zip(gold, pred)]
    results = _NLI_PIPE(inputs, truncation=True)

    scores: List[float] = []
    for res in results:
        if isinstance(res, list):
            best = max(res, key=lambda x: x.get("score", 0.0))
        else:
            best = res
        scores.append(float(best.get("score", 0.0)))
    return pa.array(scores, type=pa.float64())


# -------------------------------------------------------------------
# DuckDB registration
# -------------------------------------------------------------------

def register_metric_udfs(con: duckdb.DuckDBPyConnection) -> None:
    """
    Register all Arrow UDFs with the given DuckDB connection.
    All functions take (gold_answer, variant_response) as VARCHAR and
    return DOUBLE / BIGINT / VARCHAR.
    """
    # lexical metrics
    con.create_function(
        "exact_match_metric",
        exact_match_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "token_precision_metric",
        token_precision_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "token_recall_metric",
        token_recall_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "token_f1_metric",
        token_f1_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "rouge1_precision_metric",
        rouge1_precision_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "rouge1_recall_metric",
        rouge1_recall_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "rouge1_f1_metric",
        rouge1_f1_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "rouge2_precision_metric",
        rouge2_precision_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "rouge2_recall_metric",
        rouge2_recall_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )
    con.create_function(
        "rouge2_f1_metric",
        rouge2_f1_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "bleu_metric",
        bleu_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "edit_distance_metric",
        edit_distance_arrow,
        ["VARCHAR", "VARCHAR"],
        "BIGINT",
        type="arrow",
    )

    # embedding, BLEURT, NLI
    con.create_function(
        "embedding_cosine_similarity_metric",
        embedding_similarity_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "bleurt_metric",
        bleurt_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )

    con.create_function(
        "nli_label_metric",
        nli_label_arrow,
        ["VARCHAR", "VARCHAR"],
        "VARCHAR",
        type="arrow",
    )

    con.create_function(
        "nli_label_score_metric",
        nli_label_score_arrow,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
    )


# -------------------------------------------------------------------
# DuckDB batch mode: UPDATE table in-place (no CSV)
# -------------------------------------------------------------------

def evaluate_responses_in_duckdb(
    db_path: str,
    table_name: str = "responses",
    only_missing: bool = True,
) -> None:
    """
    Update all metric columns in-place in the DuckDB table using Arrow UDFs.

    Assumes table schema includes at least:
      gold_answer       (gold text)
      variant_response    (model response text)

    And metric columns:
        exact_match                 DOUBLE or BOOLEAN
        token_precision             DOUBLE
        token_recall                DOUBLE
        token_f1                    DOUBLE
        rouge1_precision            DOUBLE
        rouge1_recall               DOUBLE
        rouge1_f1                   DOUBLE
        rouge2_precision            DOUBLE
        rouge2_recall               DOUBLE
        rouge2_f1                   DOUBLE
        bleu                        DOUBLE
        edit_distance               BIGINT / INTEGER
        embedding_cosine_similarity DOUBLE
        bleurt                      DOUBLE
        nli_label                   VARCHAR
        nli_label_score             DOUBLE
    """
    con = duckdb.connect(db_path)
    register_metric_udfs(con)

    where_clause = "WHERE variant_response IS NOT NULL"
    if only_missing:
        where_clause += " AND exact_match IS NULL"

    # One UPDATE statement, DuckDB calls Arrow UDFs on column batches internally.
    update_sql = f"""
        UPDATE {table_name}
        SET
            exact_match                 = exact_match_metric(gold_answer, variant_response),
            token_precision             = token_precision_metric(gold_answer, variant_response),
            token_recall                = token_recall_metric(gold_answer, variant_response),
            token_f1                    = token_f1_metric(gold_answer, variant_response),
            rouge1_precision            = rouge1_precision_metric(gold_answer, variant_response),
            rouge1_recall               = rouge1_recall_metric(gold_answer, variant_response),
            rouge1_f1                   = rouge1_f1_metric(gold_answer, variant_response),
            rouge2_precision            = rouge2_precision_metric(gold_answer, variant_response),
            rouge2_recall               = rouge2_recall_metric(gold_answer, variant_response),
            rouge2_f1                   = rouge2_f1_metric(gold_answer, variant_response),
            edit_distance               = edit_distance_metric(gold_answer, variant_response),
            embedding_cosine_similarity = embedding_cosine_similarity_metric(gold_answer, variant_response),
            
        {where_clause}
    """
    """
    bleurt                        = bleu_metric(gold_answer, variant_response),
    bleurt_score                  = bleurt_metric(gold_answer, variant_response),
    nli_label                   = nli_label_metric(gold_answer, variant_response),
    nli_label_score             = nli_label_score_metric(gold_answer, variant_response)"""
    start = time.time()
    con.execute(update_sql)
    end = time.time()

    con.close()

    duration = end - start
    print(f"\nMetrics computation finished.")
    print(f"Total time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Table: '{table_name}' in DB: '{db_path}' (only_missing={only_missing})\n")


# -------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------

if __name__ == "__main__":
    DB_PATH = "llm_variants_responses.duckdb"
    TABLE_NAME = "responses"
    start_time = time.perf_counter()
    print(start_time)
    evaluate_responses_in_duckdb(
        db_path=DB_PATH,
        table_name=TABLE_NAME,
        only_missing=False,   # set False to recompute all rows
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
