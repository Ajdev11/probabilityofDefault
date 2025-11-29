Discussion of Results

1. Data Preparation and Sample Definition
The analysis uses Italian companies from AIDA. The final sample contains 48,243 unique firms. 
Default (Value=1) is flagged when legal status contains distress terms (e.g., 'fallimento', 'concordato', 'liquidazione coatta'). 
The observed class imbalance shows a default rate of approximately 2.9%. 
Outliers in financial ratios were winsorized at the 1st and 99th percentiles to limit undue influence.

2. Descriptive Statistics
Defaulted firms differ materially from healthy firms:
- Liquidity is lower for defaulted firms, pointing to tighter short‑term cash buffers.
- Solvency (equity over assets) is lower among defaulters, indicating higher leverage.
- Profitability (ROE, ROI) is weaker or negative ahead of default.
These patterns support the predictive relevance of the selected ratios (see descriptive table).

3. Econometric Results (Logistic Regression)
Liquidity ratio: Negative coefficient — higher liquidity reduces PD.
Solvency ratio: Negative coefficient — higher capitalization reduces PD.
Profitability (ROE): Negative; (ROI): Negative — higher profitability reduces PD.
The strongest single predictor by magnitude is: Solvency ratio (%) % 2017.

4. Model Validation and Performance
On the hold‑out test set (30%), the confusion matrix shows 163 true positives.
ROC‑AUC on the test set is 0.730. Interpreting AUC: 0.5=random, 1.0=perfect; this is good discrimination.
Calibration diagnostics indicate probabilistic predictions are reasonable (see reliability curve).

5. Conclusion
Financial ratios from 2017—liquidity, solvency, and profitability—are statistically meaningful predictors of default. 
The balanced logistic regression achieves good predictive performance (AUC=0.730). 
These results support its use for credit‑risk screening, with scope to improve via multi‑year features and alternative models (e.g., gradient boosting).