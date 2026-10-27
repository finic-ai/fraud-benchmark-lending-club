"""The 27 underwriter questions, the Grayson and Jev clients that ask them, and helpers.

Each decision model reads the raw application (structured fields plus the borrower's own description)
and answers plain-language questions with probabilities. Answers become log-odds features; a forward
selection on cross-validated AUC, run on the validation loans only, picks which ones to add to
XGBoost. The questions were written before any results and are not tuned on labels.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraudbench import api

FIVE = ["Very unlikely", "Unlikely", "Uncertain", "Likely", "Very likely"]

# key -> (kind, question) for yes/no; key -> (kind, question, levels) for `score` questions
POOL = {
    # Job title vs income (the raw job title is unusable for XGBoost)
    "income_plausible_for_job": ("job_income", "Is the stated annual income plausible for this job title and years of employment?"),
    "income_inflated": ("job_income", "Does the stated income look inflated for this kind of job?"),
    "job_title_real": ("job_income", "Does the job title look like a real, specific job rather than blank, vague or made up?"),
    "stable_job": ("job_income", "Does the job title suggest stable, steady employment?"),
    "income_low_for_area": ("job_income", "Is the income low relative to the cost of living in the borrower's state and ZIP area?"),
    # Description vs application
    "desc_matches_purpose": ("description", "Does the borrower's description match the stated loan purpose and title?"),
    "desc_repayment_plan": ("description", "Does the description lay out a specific, credible plan for repaying the loan?"),
    "desc_genuine": ("description", "Does the description read like a genuine personal account rather than boilerplate?"),
    "desc_evasive": ("description", "Is the description evasive, or inconsistent with the rest of the application?"),
    "hardship": ("description", "Does the borrower describe current financial hardship?"),
    # Capacity and burden
    "overextended": ("capacity", "Given income, debt-to-income ratio and revolving balance, does the borrower look financially overextended?"),
    "payment_burden": ("capacity", "Would the monthly installment be a heavy burden on this income?"),
    "loan_size_reasonable": ("capacity", "Is the loan amount reasonable for this borrower's income and purpose?"),
    "thin_history": ("capacity", "Is the credit history short or thin for someone with this job and income?"),
    "credit_seeking": ("capacity", "Does the borrower appear to be seeking a lot of credit right now?"),
    "responsible_consolidation": ("capacity", "Is this a debt consolidation by someone who appears to manage money responsibly?"),
    "risky_business": ("capacity", "Is the money going into a new or risky business venture?"),
    # Coherence
    "one_coherent_borrower": ("coherence", "Would a careful underwriter believe all these details belong to one coherent, real borrower?"),
    "mismatched_pieces": ("coherence", "Do parts of this application look mismatched, as if they don't fit the same person?"),
    # Holistic underwriting
    "underwriter_concern": ("holistic", "Would an experienced underwriter have concerns about this application?"),
    "underwriter_approve": ("holistic", "Would an experienced underwriter approve this loan at this interest rate?"),
    "grade_too_generous": ("holistic", "Does Lending Club's grade look too generous for this borrower?"),
    "will_default": ("direct", "Will the borrower default on this loan?"),
    # Graded `score` questions
    "default_risk_score": ("score", "How likely is this borrower to default on this loan?", FIVE),
    "repay_capacity_score": ("score", "How strong is this borrower's capacity to repay?",
                             ["Very weak", "Weak", "Moderate", "Strong", "Very strong"]),
    "income_plausibility_score": ("score", "How plausible is the stated income for this job title?",
                                  ["Very implausible", "Implausible", "Uncertain", "Plausible", "Very plausible"]),
    "trustworthiness_score": ("score", "Overall, how trustworthy does this application look?",
                              ["Very untrustworthy", "Untrustworthy", "Neutral", "Trustworthy", "Very trustworthy"]),
}


def _questions(questions):
    qs = {}
    for k, spec in questions.items():
        if len(spec) == 3:
            qs[k] = {"type": "score", "instructions": spec[1], "criteria": spec[2]}
        else:
            qs[k] = {"type": "noul", "instructions": spec[1]}
    return qs


def _parse(resp):
    out = {}
    for k, a in resp["answers"].items():
        if a["type"] == "score":
            out[k] = {"score": a["score"], "p": [a["probabilities"][str(i)] for i in range(len(a["probabilities"]))]}
        else:
            out[k] = a["noul"]
    return out


class Grayson(api.Grayson):
    """Grayson's documented format: structured fields in `data`, free text in `context`."""
    name = "grayson"

    def body(self, context, questions):
        body = {"model": self.model, "data": context["data"], "questions": _questions(questions)}
        if context.get("text"):
            body["context"] = context["text"]
        return body

    def parse(self, resp):
        return _parse(resp)


class Jev(api.Jev):
    """Jev takes a single `state` object, so the description goes inside it."""
    name = "jev"

    def body(self, context, questions):
        state = dict(context["data"])
        if context.get("text"):
            state["borrower_description"] = context["text"]
        return {"model": self.model, "state": state, "questions": _questions(questions)}

    def parse(self, resp):
        return _parse(resp)


def context(row, build_context):
    """Structured fields as data, the borrower's description as free text."""
    rec = build_context(row)
    return {"data": rec, "text": rec.pop("borrower_description")}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def answer_features(answers):
    """Yes/no answers -> log-odds; `score` answers -> expected level and log-odds of the top two levels."""
    cols = {}
    for k, spec in POOL.items():
        if spec[0] == "score":
            cols[k] = answers[k].map(lambda a: a["score"]).values
            cols[k + "@top2"] = logit(np.vstack(answers[k].map(lambda a: a["p"]).values)[:, -2:].sum(1))
        else:
            cols[k] = logit(answers[k].values)
    return pd.DataFrame(cols, index=answers.index)



def oof_lr(F, y, w, seed, repeats=3, C=1.0):
    """Out-of-fold predictions of a weighted L2 logistic regression, one row per repeat."""
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(F, y):
            out[r, b] = LogisticRegression(C=C, max_iter=5000).fit(F[a], y[a], sample_weight=w[a]).predict_proba(F[b])[:, 1]
    return out


def forward_select(A, y, w, seed, prefix=None, max_k=8, min_gain=0.002):
    """Forward selection on cross-validated AUC, adding one answer column at a time until it flattens."""
    def cv_auc(cols):
        F = A[cols].values if prefix is None else np.column_stack([prefix, A[cols].values])
        return float(np.mean([roc_auc_score(y, p) for p in oof_lr(F, y, w, seed)]))

    sel, cur = [], (float(np.mean([roc_auc_score(y, p) for p in oof_lr(prefix[:, None], y, w, seed)]))
                    if prefix is not None else 0.5)
    for _ in range(max_k):
        scores = {c: cv_auc(sel + [c]) for c in A.columns if c not in sel}
        best = max(scores, key=scores.get)
        if scores[best] - cur < min_gain:
            break
        sel.append(best)
        cur = scores[best]
    return sel, cur


def bad_per_10k(y, w, p, share):
    """Positives kept per 10,000 loans when only the safest `share` (by weight) is kept."""
    order = np.argsort(p)
    cw = np.cumsum(w[order]) / w.sum()
    sel = order[cw <= share]
    return 10_000 * (w[sel] * y[sel]).sum() / w.sum()


def prevalence_weights(y, true_rate):
    """Weight the negative rows so the weighted positive share equals true_rate."""
    y = np.asarray(y)
    n1, n0 = y.sum(), len(y) - y.sum()
    return np.where(y == 1, 1.0, n1 * (1 - true_rate) / (n0 * true_rate))
