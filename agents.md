# AGENTS.md - Repository Guidelines for Jules

- Stack: Python 3.10+, PyTorch, PySpark (if spark pipelines present), Scikit-learn, Streamlit, Pandas
- Objective: Replace/upgrade existing baseline model with a deep neural network tailored for extreme class imbalance.
- Evaluation Criteria: Prioritize PR-AUC (Area Under Precision-Recall Curve), Recall at 95%+ Precision, and F1-score over raw accuracy.
- Interface: Provide a lightweight Streamlit app allowing CSV upload and anomaly batch scoring.
- Testing: All scripts and pipeline endpoints must include unit tests runnable via `pytest`.
