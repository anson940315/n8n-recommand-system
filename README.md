# 104 B2B Lead Scoring and Outreach

This project ranks potential B2B leads by similarity to existing customers, then prepares a shortlist for outreach. It is designed for local private datasets and optional n8n automation.

## What It Does

- Loads existing customer profiles and potential company profiles
- Normalizes company, welfare, location, industry, and size metadata
- Scores candidates with weighted cosine-similarity features
- Excludes existing customers and recently developed companies
- Exports ranked lead lists for review or downstream automation
- Provides a Flask API for background job orchestration

## Data Privacy

Real customer data, vendor-provided files, generated lead lists, n8n exports, Google Sheets URLs, Gmail settings, and credentials must not be committed. Keep those files under `private/` or another local-only location.

This repository includes only synthetic sample data under `examples/` so the pipeline can be demonstrated without exposing business data.

## Repository Layout

```text
.
├── api_server.py
├── company_104_client.py
├── cosine_similarity_analysis.py
├── email_enrichment.py
├── random_104_pipeline.py
├── examples/
│   ├── sample_customers.csv
│   └── sample_potentials.csv
├── outputs/
│   └── .gitkeep
├── private/
│   └── README.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p outputs private
```

## Run With Synthetic Data

```bash
python3 cosine_similarity_analysis.py \
  --customers examples/sample_customers.csv \
  --potentials examples/sample_potentials.csv \
  --output-dir outputs/demo \
  --top-leads-file outputs/demo/top_leads.csv \
  --top-n 5
```

The command writes ranking files and top leads under `outputs/demo/`. The `outputs/` folder is ignored by Git.

## Run With Private Data

Place real files locally, for example:

```text
private/customer_profiles.xlsx
private/random_104_gcis_filtered_profiles.xlsx
```

Then run:

```bash
python3 cosine_similarity_analysis.py
```

You can also crawl fresh potential companies before scoring:

```bash
python3 cosine_similarity_analysis.py --crawl-before-score
```

## API Server

```bash
python3 api_server.py
```

Health check:

```bash
curl http://127.0.0.1:5001/health
```

The API writes job state and outputs under `outputs/`, which is intentionally excluded from Git.

## Public Sharing Checklist

Before publishing:

- Confirm `git status --ignored` shows real `.xlsx`, `.csv`, `.json`, logs, and `outputs/` as ignored
- Do not publish n8n workflow exports unless they are fully sanitized
- Do not publish generated lead lists from live runs
- Use `examples/` for demos and screenshots

## Limitations

- The scoring model is heuristic and should be treated as decision support, not an automated sales decision.
- 104 and public website crawling can fail due to network changes, rate limits, or site markup changes.
- Outreach workflows should be reviewed for consent, compliance, and sender reputation before production use.
