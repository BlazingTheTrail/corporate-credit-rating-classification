# Data

This project uses the public Kaggle dataset **Corporate Credit Rating With
Financial Ratios**, published by Kirtan Delwadia under CC BY 4.0.

Dataset page:
<https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios>

The raw CSV is intentionally excluded from Git. Download it with:

```bash
python scripts/download_data.py
```

Expected location:

```text
data/raw/corporateCreditRatingWithFinancialRatios.csv
```

The dataset covers historical ratings from 2010 through 2016. It is suitable
for a public modeling benchmark, but it is not a substitute for current
issuer-level credit analysis or a production rating methodology.
