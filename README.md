# AI-Assist

A lightweight analysis toolkit for Retrieval-Augmented Generation (RAG) chatbots.  
It helps you 🔍 inspect log files, validate key quality hypotheses, and generate concise PDF reports for stakeholders.

## What’s inside

| File | Purpose |
|------|---------|
| **`gen_analysis.py`** | Loads raw interaction logs, computes latency/feedback/source stats, and prints an at-a-glance console report. |
| **`hypotheses.py`**  | Five statistical tests (latency→feedback, source quality, etc.) packaged as a single CLI script. |
| **`Report - Davit Davtyan.pdf`** | Full Report |
| **`.gitignore`** | Standard Python ignores. |
| **`README.md`** | You’re reading it 🙂 |

## Quick start

```bash
# create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# run general stats
python gen_analysis.py logs.json

# run hypothesis validation
python hypotheses.py logs.json
