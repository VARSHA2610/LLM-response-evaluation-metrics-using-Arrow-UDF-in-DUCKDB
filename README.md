# LLM-response-evaluation-metrics-using-Arrow-UDF-in-DUCKDB

"""
Compute LLM evaluation metrics between a gold answer (base_question)
and a model response (variant_response).

Metrics:
- Exact Match
- Token Precision / Recall / F1
- ROUGE-1 / ROUGE-2 (P/R/F1)
- BLEU
- Edit Distance (Levenshtein)
- Embedding cosine similarity (SentenceTransformer)
- BLEURT (via Hugging Face Elron/bleurt-base-512)
- NLI label (entailment / neutral / contradiction) via BART MNLI

Supports:
- Batch evaluation from a DuckDB file, updating the table in-place.
"""