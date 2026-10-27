"""Loading, the time split, sampling, XGBoost features and API contexts for Lending Club loans.

Task: for a completed loan whose application has a borrower description, will it be charged off?
Only fields known at application time are used. Payment history, recoveries, last FICO pull and
every other post-origination column are excluded to avoid leakage. Lending Club's own grade and
interest rate are kept, as in most papers; they are its risk model's output at origination.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
DATASET = "wordsforthewise/lending-club"
LABEL = "charged_off"
DATE = "issue_d"

# Split by issue month: the earliest ~70% of loans train, the next ~10% validate, the last ~20% test.
TRAIN_SHARE, VAL_SHARE = 0.70, 0.10

APPLICATION_FIELDS = [
    "id", "issue_d", "loan_status", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc", "verification_status", "purpose", "title",
    "zip_code", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status", "application_type", "desc",
]
FINAL_STATUS = {
    "Fully Paid": 0, "Charged Off": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}
NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
           "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
           "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "zip_code", "addr_state",
               "initial_list_status", "application_type"]
GRADES = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]


def download_raw(raw_dir=RAW_DIR):
    """Fetch the accepted-loans file with kagglehub. Needs Kaggle credentials (KAGGLE_API_TOKEN in
    .env, ~/.kaggle/access_token or ~/.kaggle/kaggle.json)."""
    raw_dir = Path(raw_dir)
    found = sorted(raw_dir.glob("**/accepted_2007_to_2018*.csv*"))
    if not found:
        import kagglehub

        kagglehub.dataset_download(DATASET, output_dir=str(raw_dir))
        found = sorted(raw_dir.glob("**/accepted_2007_to_2018*.csv*"))
    # Prefer the plain CSV when both it and the .gz are present.
    return min(found, key=lambda p: p.suffix == ".gz")


_DESC_PREFIX = re.compile(r"Borrower added on \d\d/\d\d/\d\d >\s*", re.I)


def clean_desc(s):
    if pd.isna(s):
        return ""
    s = _DESC_PREFIX.sub(" ", str(s)).replace("<br>", " ").replace("<br/>", " ")
    return re.sub(r"\s+", " ", s).strip()


def load(raw_dir=RAW_DIR):
    """Completed loans with a non-empty borrower description, cached as parquet."""
    raw_dir = Path(raw_dir)
    cache = raw_dir / "loans_with_desc.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(download_raw(raw_dir), usecols=APPLICATION_FIELDS, dtype=str)
    df = df[df["loan_status"].isin(FINAL_STATUS)].copy()
    df["desc"] = df["desc"].map(clean_desc)
    df = df[df["desc"].str.len() > 0].copy()
    df[LABEL] = df.pop("loan_status").map(FINAL_STATUS).astype(int)
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c].str.rstrip("%"), errors="coerce")
    df["term_months"] = pd.to_numeric(df.pop("term").str.extract(r"(\d+)")[0])
    df["emp_length_years"] = pd.to_numeric(
        df.pop("emp_length").str.replace("< 1", "0").str.extract(r"(\d+)")[0])
    df["id"] = df["id"].astype("int64")
    df = df.sort_values(["issue_d", "id"]).reset_index(drop=True)
    df.to_parquet(cache)
    return df


def assign_split(df):
    """Whole issue months only, so no month straddles two splits."""
    share = df.groupby("issue_d").size().sort_index().cumsum() / len(df)
    month_split = pd.Series(
        np.select([share <= TRAIN_SHARE, share <= TRAIN_SHARE + VAL_SHARE], ["train", "val"], "test"),
        index=share.index)
    df["split"] = df["issue_d"].map(month_split)
    return df


def stratified_sample(df, n_fraud, n_total, seed):
    """Fixed-size sample with a set number of positive rows; returns sorted ids."""
    pos = df[df[LABEL] == 1].sample(n=n_fraud, random_state=seed)
    neg = df[df[LABEL] == 0].sample(n=n_total - n_fraud, random_state=seed)
    return np.sort(np.concatenate([pos["id"].values, neg["id"].values]))


def text(df):
    """Borrower description, loan title and job title: everything the borrower wrote."""
    return (df["title"].fillna("") + ". " + df["emp_title"].fillna("") + ". " + df["desc"]).str.strip(". ")


def tabular_features(df):
    X = df[NUMERIC + ["term_months", "emp_length_years"]].astype("float32").copy()
    X["sub_grade"] = df["sub_grade"].map({g: i for i, g in enumerate(GRADES)}).astype("float32")
    X["credit_history_months"] = ((df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44).astype("float32")
    for c in CATEGORICAL:
        codes, _ = pd.factorize(df[c], sort=True)
        X[c] = np.where(codes < 0, np.nan, codes).astype("float32")
    X["emp_title_freq"] = df["emp_title"].str.lower().map(
        df["emp_title"].str.lower().value_counts(normalize=True)).astype("float32")
    X["desc_chars"] = df["desc"].str.len().astype("float32")
    return X


def _num(v, nd=2):
    return None if pd.isna(v) else round(float(v), nd)


def _str(v):
    return None if pd.isna(v) else str(v)


def build_context(row):
    return {
        "record_type": "personal loan application on Lending Club",
        "issue_month": row["issue_d"].strftime("%Y-%m"),
        "loan": {
            "amount_usd": _num(row["loan_amnt"], 0), "term_months": _num(row["term_months"], 0),
            "interest_rate_pct": _num(row["int_rate"]), "monthly_installment_usd": _num(row["installment"]),
            "lending_club_sub_grade": _str(row["sub_grade"]), "purpose": _str(row["purpose"]),
            "title": _str(row["title"]),
        },
        "borrower": {
            "job_title": _str(row["emp_title"]), "years_employed": _num(row["emp_length_years"], 0),
            "home_ownership": _str(row["home_ownership"]), "annual_income_usd": _num(row["annual_inc"], 0),
            "income_verification": _str(row["verification_status"]), "state": _str(row["addr_state"]),
            "zip3": _str(row["zip_code"]), "debt_to_income_pct": _num(row["dti"]),
        },
        "credit": {
            "fico_range": [_num(row["fico_range_low"], 0), _num(row["fico_range_high"], 0)],
            "earliest_credit_line": None if pd.isna(row["earliest_cr_line"]) else row["earliest_cr_line"].strftime("%Y-%m"),
            "delinquencies_last_2y": _num(row["delinq_2yrs"], 0),
            "inquiries_last_6m": _num(row["inq_last_6mths"], 0),
            "months_since_last_delinquency": _num(row["mths_since_last_delinq"], 0),
            "public_records": _num(row["pub_rec"], 0), "open_accounts": _num(row["open_acc"], 0),
            "total_accounts": _num(row["total_acc"], 0), "revolving_balance_usd": _num(row["revol_bal"], 0),
            "revolving_utilization_pct": _num(row["revol_util"]),
        },
        "borrower_description": row["desc"],
    }
