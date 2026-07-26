# Data Dictionary

## Source identifiers and metadata

| Field | Role in project | Model input |
|---|---|---|
| Rating Agency | Agency issuing the observed rating | Yes |
| Corporation | Issuer name | No |
| CIK | SEC issuer identifier; used for grouped validation | No |
| Ticker | Exchange ticker | No |
| Rating Date | Observation date; used for temporal splitting | No |
| SIC Code | Standard Industrial Classification code | Yes, categorical |
| Sector | Broad industry sector | Yes, categorical |

## Targets

| Field | Definition | Model input |
|---|---|---|
| Rating | Original 23-notch source rating | No |
| Binary Rating | Source investment-grade indicator derived from Rating | No |
| target_binary | Reconstructed investment/speculative-grade target | No |
| target_7band | AAA, AA, A, BBB, BB, B, or CCC-or-lower | No |

`Binary Rating` is intentionally excluded because it reveals the broad target
class and would create direct target leakage.

## Financial-ratio predictors

| Field |
|---|
| Current Ratio |
| Long-term Debt / Capital |
| Debt/Equity Ratio |
| Gross Margin |
| Operating Margin |
| EBIT Margin |
| EBITDA Margin |
| Pre-Tax Profit Margin |
| Net Profit Margin |
| Asset Turnover |
| ROE - Return On Equity |
| Return On Tangible Equity |
| ROA - Return On Assets |
| ROI - Return On Investment |
| Operating Cash Flow Per Share |
| Free Cash Flow Per Share |

All preprocessing is learned within the model pipeline. The project does not
replace extreme values globally because unusual financial ratios may contain
credit-risk information.
