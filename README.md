# 🤖 LLM-response-evaluation-metrics-using-Arrow-UDF-in-DUCKDB

## 📌 Overview

Compute LLM evaluation metrics between a gold answer (`base_question`)  
and a model-generated response (`variant_response`).

This project provides a scalable evaluation pipeline using DuckDB, Apache Arrow UDFs, and NLP evaluation models to benchmark Large Language Model (LLM) responses efficiently in batch mode.

---

# 📊 Supported Evaluation Metrics

- ✅ Exact Match
- 🎯 Token Precision / Recall / F1
- 📝 ROUGE-1 / ROUGE-2 (Precision / Recall / F1)
- 📚 BLEU Score
- ✏️ Edit Distance (Levenshtein Distance)
- 🧠 Embedding Cosine Similarity (`SentenceTransformer`)
- 📈 BLEURT (`Elron/bleurt-base-512`)
- 🔍 Natural Language Inference (NLI)
  - Entailment
  - Neutral
  - Contradiction
  via BART MNLI

---

# 🏗️ Components of the System

## 🗂️ LLM Response Dataset in DuckDB Table

Stores:
- Base question
- Gold answer
- Variant categories
- Model responses
- Model versions
- Evaluation metric values

---

## ⚡ Arrow-Based UDF Layer

A collection of vectorized UDFs registered in DuckDB to compute evaluation metrics efficiently in batches using Apache Arrow.

---

## 📐 Metric Computation Modules

Implements:
- Statistical scorers
- Embedding-based scorers
- Transformer-based evaluation models

---

## 🧮 Analytical Queries in SQL

Uses SQL queries to:
- Aggregate evaluation scores
- Compare models
- Analyze response variants
- Generate benchmarking insights

---

# 🔄 Components Interaction

- Python scripts fetch responses from LLM models for each question and store them in DuckDB tables.
- SQL queries invoke Arrow-based UDFs on relevant columns when metrics are computed.
- DuckDB passes data to UDFs as Apache Arrow batches.
- Each UDF computes evaluation metrics vectorized across the batch.
- Results are written back into DuckDB result tables.
- SQL analytics aggregate metrics across models, response variants, and categories.

---

# 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core backend and metric computation |
| Database Engine | DuckDB | Analytical database for evaluation workflows |
| Vectorized Processing | Apache Arrow | High-performance batch data transfer |
| NLP Models | Hugging Face Transformers | BLEURT and NLI scoring |
| Embedding Models | SentenceTransformers | Semantic similarity computation |
| Evaluation Metrics | ROUGE, BLEU, Levenshtein | Text similarity and scoring |
| Query Engine | SQL | Aggregation and analytical evaluation |
| Data Processing | Pandas / PyArrow | Data manipulation and Arrow integration |

---

# 🚀 Key Features

- Batch LLM response evaluation
- Arrow-based vectorized metric computation
- SQL-driven analytics workflows
- Embedding similarity scoring
- Transformer-based semantic evaluation
- Scalable DuckDB integration
- Efficient in-place metric updates
- Multi-model benchmarking support

---

# 🌐 Workflow

```text
LLM Responses
      ↓
DuckDB Tables
      ↓
Arrow-Based UDFs
      ↓
Metric Computation
      ↓
Result Tables
      ↓
SQL Analytics & Benchmarking
```

---

# 🎯 Outcomes

- Benchmark multiple LLM models efficiently
- Compare semantic quality of responses
- Analyze robustness across response variants
- Enable scalable evaluation pipelines
- Generate analytical insights using SQL
- Improve reliability and consistency of LLM outputs

---
