# data/ — not distributed

This directory is intentionally empty in the repository. The sampled
documents and teacher-labeled training sets are derived from third-party
corpora (Enron email archive, Hugging Face datasets, Apache JIRA, public
chat archives, GitHub code, LogHub) whose licenses do not permit
redistribution here.

Rebuild locally with `data_prep/sample_2k.py` (point it at corpora you have
rights to), then `data_prep/build_gpt_batches.py` → `harvest_gpt.py` →
`build_collie_sft.py`. See the main README for the synthetic/augmented-data
alternative.
