# Lending Club: Does a Decision Model Help XGBoost Predict Loan Default?

XGBoost is the standard model for credit risk. We tested whether it gets better at predicting which
personal loans will be charged off when it also gets a decision model's answers to plain-language
underwriter questions about each application. We compared **Grayson** (Finic) and **Jev** (TypeSafe).

Write-up: [finic.ai/blog/decision-models-xgboost-loan-default](https://finic.ai/blog/decision-models-xgboost-loan-default)

## Results

| Approach | ROC-AUC | vs XGBoost |
| --- | --- | --- |
| XGBoost | 0.681 | — |
| TF-IDF (borrower's text only) | 0.582 | −0.099 [−0.131, −0.064] |
| XGBoost + TF-IDF | 0.684 | +0.004 [−0.003, +0.010] |
| XGBoost + Jev | 0.687 | +0.006 [−0.003, +0.016] |
| **XGBoost + Grayson** | **0.689** | **+0.008 [+0.001, +0.016]** |

ROC-AUC is 0.5 for random guessing and 1.0 for a perfect ranking. Scores are on 3,000 loans issued
after every loan used for fitting; brackets are 95% bootstrap intervals.

Every loan in this data was one Lending Club approved. Suppose the lender had also used the model to
decline the riskiest 10% of them. A better ranking means more of the declined loans are ones that would
have been charged off, and fewer are good loans turned away. Using the test period's actual loan
outcomes ($5,114 average loss on a charged-off loan, $3,207 average profit on a repaid one):

| Approach | Fewer defaults per 10,000 loans | Gain per 10,000 loans |
| --- | --- | --- |
| XGBoost + TF-IDF | 6 | +$52,000 |
| XGBoost + Jev | 12 | +$103,000 |
| **XGBoost + Grayson** | **19** | **+$155,000** |

Results when declining the riskiest 20%, 30% and 50% are in `results/roi.csv`.

## Why Grayson helps

Many Lending Club borrowers explained in their own words why they needed the loan. XGBoost can't
read that text; it only sees how long it is. Grayson reads it. Of the 27 questions we asked, the one
that added the most to XGBoost was "Does the borrower describe current financial hardship?", an
answer that comes straight from the borrower's description. Jev was asked the same question, but its
answers added less. A traditional text model (TF-IDF) can pick up which words go with default, but it
added only about half as much to XGBoost as Grayson did.

## How it works

1. **Data.** Lending Club's public accepted-loans file: about 126,000 completed loans from 2007 to
   2016 whose application includes a borrower description. A loan counts as a default if it was
   charged off.
2. **Time split.** Loans are split by issue month: the earliest 70% train XGBoost, the next 10% are
   the validation period and the latest 20% are the test period.
3. **Questions.** 27 underwriter questions, such as whether the income is plausible for the job title
   or whether the description matches the loan's purpose, written before any results.
4. **Answers.** 2,000 validation loans and 3,000 test loans go to Grayson and Jev once each, with the
   application fields and the borrower's description. Neither model sees whether the loan was repaid.
5. **Models.** For each decision model, the answers that add most to the XGBoost score are selected on
   the validation loans and combined with it in a small logistic layer, then scored once on the test
   loans. For comparison, a TF-IDF text model trained on the borrowers' words is combined with XGBoost
   the same way.

## Run it yourself

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # macOS: brew install libomp
cp .env.example .env    # add KAGGLE_API_TOKEN (kaggle.com -> Settings -> API) to download the loan data
.venv/bin/jupyter nbconvert --to notebook --execute --inplace benchmark.ipynb
```

The notebook downloads Lending Club's loan file (about 1.6 GB) from Kaggle on first run. The Grayson
and Jev answers are cached in `data/api_responses/`, so no API keys are needed. To query the APIs
yourself, also add `GRAYSON_API_KEY` and `TYPESAFE_API_KEY` to `.env`.

## Files

```
benchmark.ipynb      the whole benchmark
fraudbench/          data loading, API clients and the 27 questions
data/api_responses/  cached Grayson and Jev answers
results/             results tables and ROC curve
```
