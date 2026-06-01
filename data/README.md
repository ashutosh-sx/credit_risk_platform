# 📂 Home Credit Dataset Directory

This directory is designated for raw and processed datasets used by the Credit Risk Platform.

> [!NOTE]
> Files in this directory are ignored by Git (except this `README.md`) to avoid committing massive CSV files to version control. They are mounted directly inside the Docker containers when running containerized workloads.

## 📥 Required Dataset Structure

For the loader and training pipeline to function, place your downloaded dataset files (e.g., from the Kaggle Home Credit Default Risk competition) here:

* `application_train.csv` - Main training table containing applicants' demographic, personal, and financial data.
* `application_test.csv` - Main testing table.
* `bureau.csv` - Monthly credit bureau histories of applicants.
* `bureau_balance.csv` - Balance sheets of applicant histories.
* `POS_CASH_balance.csv` - Monthly cash balance sheets.
* `credit_card_balance.csv` - Monthly credit card balance sheets.
* `previous_application.csv` - Previous applications of clients in the system.
* `installments_payments.csv` - Repayment histories for previous credits.

## 💾 Local SQL Database
If using SQLite or DuckDB for the "Talk to Data" feature:
* Place the database file `credit_risk.db` (or similar configured in `.env`) in this directory.
