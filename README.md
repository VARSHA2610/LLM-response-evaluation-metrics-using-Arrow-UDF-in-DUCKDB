# LLM-response-evaluation-metrics-using-Arrow-UDF-in-DUCKDB


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

Component of the System

- LLM Response Dataset in DuckDB table: Stores base question, gold
answer, variant categories, model responses, model versions and metrics
value.
- Arrow-Based UDF Layer: A collection of vectorised UDFs registered in
DuckDB to compute evaluation metrics in batches using Apache Arrow.
- Metric Computation Modules: Implements statistical scorers and models-
based scorers as metrics.
- Analytical queries in SQL: Uses SQL to aggregate, group, and analyze
evaluation results.

 Components Interaction :
- Python script fetches response from LLM model for each question and stores
them in DuckDB table.
- SQL queries invoke Arrow-based UDFs on relevant columns when metrics
are called
- DuckDB passes data to UDFs as Arrow batches.
- Each UDF computes metrics vectorised over the batch.
- Results are written back to DuckDB result tables.
- SQL aggregates metrics across models, variants, and categories.
