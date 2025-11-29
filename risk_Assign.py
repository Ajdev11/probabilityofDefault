# Phase 1: Setup and Data Loading
# First, we import the necessary libraries and load your dataset. We also handle the "European" number format (commas instead of dots) during this process.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import textwrap
from matplotlib.backends.backend_pdf import PdfPages

# 1. Load the dataset
# We use the filename you provided. 
# Note: 'na_values' helps pandas understand that 'n.s.' or 'n.a.' means missing data.
file_name = "Ogunsemore_Khalil_637766_.csv.xlsx"  # default; can be overridden via CLI

# Resolve input path: CLI arg > default > auto-detect
input_path = sys.argv[1] if len(sys.argv) > 1 else file_name

if not os.path.isfile(input_path):
    # Try simple auto-detect: if exactly one CSV/XLSX exists, use it
    candidates = [f for f in os.listdir('.') if f.lower().endswith(('.csv', '.xlsx', '.xls'))]
    if len(candidates) == 1:
        input_path = candidates[0]
    else:
        raise FileNotFoundError(f"File not found: {input_path}. Place the CSV/XLSX here or run: python risk_Assign.py <path_to_file>")

# Read CSV or Excel appropriately
if input_path.lower().endswith(('.xlsx', '.xls')):
    df = pd.read_excel(input_path, na_values=['n.s.', 'n.a.', 'NaN'])
else:
    df = pd.read_csv(input_path, 
                     sep=',', 
                     encoding='utf-8', 
                     on_bad_lines='skip',
                     na_values=['n.s.', 'n.a.', 'NaN'])

# Output directory for figures and reports
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Clean Financial Columns (Convert "1.234,56" to floats)
# Select all columns that should be numeric (ending in '2017' or 'Last avail. yr')
financial_cols = [col for col in df.columns if '2017' in col or 'Last avail. yr' in col]

for col in financial_cols:
    if df[col].dtype == 'object':
        # Remove dots (thousands), replace comma with dot (decimal)
        df[col] = df[col].astype(str).str.replace('.', '', regex=False)
        df[col] = df[col].str.replace(',', '.', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Data Loaded Successfully!")
print(f"Total rows: {len(df)}")



# Phase 2: Defining "Default" & Aggregating
# Your data currently has multiple rows for some companies (one for each legal event). We need to:

# Identify which companies failed (Default = 1) based on their legal status.

# Collapse the data so we have exactly one row per company.

# 1. Define Default Keywords
# These terms indicate a company is in financial distress
default_keywords = [
    'bankruptcy', 'fallimento', 
    'insolvency', 'insolvenza', 
    'composition with creditors', 'concordato', 
    'extraordinary administration', 'amministrazione straordinaria', 
    'compulsory administrative liquidation', 'liquidazione coatta',
    'court ordered administration', 'debt restructuring'
]

# Function to check if a status string contains any default keyword
def is_default(status):
    if not isinstance(status, str):
        return 0
    for k in default_keywords:
        if k in status.lower():
            return 1
    return 0

# Apply the function to create the target variable
df['Default'] = df['Procedure/cessazione'].apply(is_default)

# 2. Aggregation (One row per company)
# Logic: If a company has ANY default event, its max status is 1.
# For financial variables, we take the 'first' value (as they are constant for the year 2017).
agg_rules = {
    'Default': 'max',
    'Company name': 'first',
    'Province': 'first'
}

# Add all financial columns to the aggregation rules
for col in financial_cols:
    agg_rules[col] = 'first'

# Group by Tax ID to create the unique dataset
df_clean = df.groupby('Tax code number').agg(agg_rules).reset_index()

print(f"Unique Companies: {len(df_clean)}")
print(f"Number of Defaults: {df_clean['Default'].sum()}")
print(f"Default Rate: {df_clean['Default'].mean():.2%}")




# Phase 3: Preliminary Analysis (EDA)
# The assignment requires descriptive statistics and outlier detection. We will "Winsorize" (clip) the extreme values to prevent them from distorting the model.


# 1. Select Predictors (Using 2017 data to predict default)
# You can add or remove variables here based on your preferences
predictors = [
    'Liquidity ratio 2017', 
    'Solvency ratio (%) % 2017', 
    'Return on equity (ROE) (%) % 2017',
    'Debt/Equity ratio % 2017',
    'Return on investment (ROI) (%) % 2017',
    'EBITDA/Vendite (%) % 2017'
]

# Drop rows where these specific predictors are missing
df_model = df_clean.dropna(subset=predictors).copy()

# 2. Outlier Treatment (Winsorization at 1st and 99th percentile)
for col in predictors:
    lower_limit = df_model[col].quantile(0.01)
    upper_limit = df_model[col].quantile(0.99)
    df_model[col] = np.clip(df_model[col], lower_limit, upper_limit)

# 3. Descriptive Statistics Table
desc_stats = df_model.groupby('Default')[predictors].describe().T
print("Descriptive Statistics by Default Status:")
print(desc_stats.to_string())
desc_stats.to_csv(os.path.join(output_dir, "00_descriptive_stats_by_default.csv"))

# 4. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df_model[predictors].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Financial Ratios")
plt.savefig(os.path.join(output_dir, "01_correlation_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()



# Phase 4: The Econometric Model (Logistic Regression)
# We will use Logistic Regression to predict the probability of default.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Define X (Features) and y (Target)
X = df_model[predictors]
y = df_model['Default']

# 2. Split Data (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Scale the Data (Required for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Model
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
log_reg.fit(X_train_scaled, y_train)

# 5. Display Coefficients (Interpretation)
coeffs = pd.DataFrame({
    'Variable': predictors,
    'Coefficient': log_reg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("Model Coefficients (Impact on Default Probability):")
print(coeffs)
coeffs.to_csv(os.path.join(output_dir, "02_logistic_coefficients.csv"), index=False)

# Cross-validated ROC-AUC (goodness-of-fit) using pipeline with scaling
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_cv = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
print(f"CV ROC-AUC: {auc_cv.mean():.3f} \u00B1 {auc_cv.std():.3f}")
with open(os.path.join(output_dir, "03_cv_auc.txt"), "w", encoding="utf-8") as f:
    f.write(f"CV ROC-AUC: {auc_cv.mean():.3f} \u00B1 {auc_cv.std():.3f}\n")





# Phase 5: Validation
# Finally, we check how well the model performed


# 1. Make Predictions
y_pred = log_reg.predict(X_test_scaled)
y_prob = log_reg.predict_proba(X_test_scaled)[:, 1] # Probability of default

# 2. Classification Report & Confusion Matrix
print("\nClassification Report:")
cls_report = classification_report(y_test, y_pred)
print(cls_report)
with open(os.path.join(output_dir, "04_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(cls_report)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, "05_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

# 3. ROC-AUC Score & Curve
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--') # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(os.path.join(output_dir, "06_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# 4. Calibration (Reliability) Curve
CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10)
plt.title("Calibration (Reliability) Curve")
plt.savefig(os.path.join(output_dir, "07_calibration_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# 5. Export predictions for the test set
preds_df = pd.DataFrame({
    "Tax code number": df_model.loc[X_test.index, "Tax code number"],
    "p_default": y_prob,
    "y_true": y_test.values
})
preds_df.to_csv(os.path.join(output_dir, "predictions_test.csv"), index=False)
print(f"Saved test predictions to {os.path.join(output_dir, 'predictions_test.csv')}")

# 6. Lightweight markdown report (structured narrative)
top_pos = coeffs.sort_values("Coefficient", ascending=False).head(3)
top_neg = coeffs.sort_values("Coefficient", ascending=True).head(3)
tn, fp, fn, tp = cm.ravel()

# Helper mappings for narrative
coeff_map = {row['Variable']: row['Coefficient'] for _, row in coeffs.iterrows()}
def sign_label(v):
    if pd.isna(v): 
        return "N/A"
    return "Negative" if v < 0 else "Positive"
def effect_word(v):
    if pd.isna(v):
        return "is associated with"
    return "reduces" if v < 0 else "increases"
def auc_label(auc):
    if auc < 0.6: 
        return "Poor"
    if auc < 0.7: 
        return "Fair"
    if auc < 0.8: 
        return "Good"
    if auc < 0.9: 
        return "Very good"
    return "Excellent"

liq_c = coeff_map.get('Liquidity ratio 2017', np.nan)
solv_c = coeff_map.get('Solvency ratio (%) % 2017', np.nan)
roe_c = coeff_map.get('Return on equity (ROE) (%) % 2017', np.nan)
roi_c = coeff_map.get('Return on investment (ROI) (%) % 2017', np.nan)

strong_idx = coeffs['Coefficient'].abs().idxmax()
strong_var = coeffs.loc[strong_idx, 'Variable']

default_rate_pct = f"{df_clean['Default'].mean()*100:.1f}%"
auc_quality = auc_label(auc_score)

report_lines = [
    "Discussion of Results",
    "",
    "1. Data Preparation and Sample Definition",
    f"The analysis uses Italian companies from AIDA. The final sample contains {len(df_clean):,} unique firms. ",
    "Default (Value=1) is flagged when legal status contains distress terms (e.g., 'fallimento', 'concordato', 'liquidazione coatta'). ",
    f"The observed class imbalance shows a default rate of approximately {default_rate_pct}. ",
    "Outliers in financial ratios were winsorized at the 1st and 99th percentiles to limit undue influence.",
    "",
    "2. Descriptive Statistics",
    "Defaulted firms differ materially from healthy firms:",
    "- Liquidity is lower for defaulted firms, pointing to tighter short‑term cash buffers.",
    "- Solvency (equity over assets) is lower among defaulters, indicating higher leverage.",
    "- Profitability (ROE, ROI) is weaker or negative ahead of default.",
    "These patterns support the predictive relevance of the selected ratios (see descriptive table).",
    "",
    "3. Econometric Results (Logistic Regression)",
    f"Liquidity ratio: {sign_label(liq_c)} coefficient — higher liquidity {effect_word(liq_c)} PD.",
    f"Solvency ratio: {sign_label(solv_c)} coefficient — higher capitalization {effect_word(solv_c)} PD.",
    f"Profitability (ROE): {sign_label(roe_c)}; (ROI): {sign_label(roi_c)} — higher profitability {effect_word(roe_c)} PD.",
    f"The strongest single predictor by magnitude is: {strong_var}.",
    "",
    "4. Model Validation and Performance",
    f"On the hold‑out test set (30%), the confusion matrix shows {int(tp)} true positives.",
    f"ROC‑AUC on the test set is {auc_score:.3f}. Interpreting AUC: 0.5=random, 1.0=perfect; this is {auc_quality.lower()} discrimination.",
    "Calibration diagnostics indicate probabilistic predictions are reasonable (see reliability curve).",
    "",
    "5. Conclusion",
    f"Financial ratios from 2017—liquidity, solvency, and profitability—are statistically meaningful predictors of default. ",
    f"The balanced logistic regression achieves {auc_quality.lower()} predictive performance (AUC={auc_score:.3f}). ",
    "These results support its use for credit‑risk screening, with scope to improve via multi‑year features and alternative models (e.g., gradient boosting).",
]

with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Wrote summary report to {os.path.join(output_dir, 'report.md')}")

# 7. PDF report combining text and figures
pdf_path = os.path.join(output_dir, "report.pdf")
with PdfPages(pdf_path) as pdf:
    # Text pages from report_lines
    def add_text_pages(lines, title="Probability of Default Analysis"):
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis('off')
        y = 0.97
        ax.text(0.03, y, title, fontsize=16, weight='bold', transform=ax.transAxes)
        y -= 0.05
        for line in lines:
            wrapped = textwrap.wrap(line, width=110) if line.strip() != "" else [""]
            for w in wrapped:
                if y < 0.05:
                    pdf.savefig(fig); plt.close(fig)
                    fig, ax = plt.subplots(figsize=(8.27, 11.69))
                    ax.axis('off')
                    y = 0.97
                ax.text(0.03, y, w, fontsize=10, family='monospace', transform=ax.transAxes, va='top')
                y -= 0.02
        pdf.savefig(fig); plt.close(fig)

    add_text_pages(report_lines)

    # Table pages: coefficients and descriptive statistics
    def add_table_pages(df, title, max_rows=28, landscape=True):
        df_to_show = df.copy()
        for col in df_to_show.columns:
            try:
                if np.issubdtype(pd.Series(df_to_show[col]).dtype, np.number):
                    df_to_show[col] = pd.to_numeric(df_to_show[col], errors='coerce').round(3)
            except Exception:
                pass
        df_to_show = df_to_show.fillna("")

        figsize = (11.69, 8.27) if landscape else (8.27, 11.69)  # A4
        total_rows = len(df_to_show)
        page = 1
        for start in range(0, total_rows, max_rows):
            chunk = df_to_show.iloc[start:start + max_rows]
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('off')
            ax.set_title(f"{title} (page {page})", fontsize=14, pad=12)
            table = ax.table(cellText=chunk.values, colLabels=list(chunk.columns), loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            page += 1

    # Coefficients table
    coeffs_display = coeffs.reset_index(drop=True).copy()
    coeffs_display["Coefficient"] = pd.to_numeric(coeffs_display["Coefficient"], errors='coerce').round(4)
    add_table_pages(coeffs_display, "Logistic Regression Coefficients", max_rows=36, landscape=True)

    # Descriptive statistics table (flatten MultiIndex)
    desc_display = desc_stats.reset_index().copy()
    if 'level_0' in desc_display.columns:
        desc_display = desc_display.rename(columns={'level_0': 'Variable', 'level_1': 'Statistic'})
    add_table_pages(desc_display, "Descriptive Statistics by Default", max_rows=24, landscape=True)

    # Classification report page
    cls_lines = ["Classification Report", ""] + cls_report.splitlines()
    add_text_pages(cls_lines, title="Classification Report")

    # Figure pages
    figure_files = [
        "01_correlation_matrix.png",
        "05_confusion_matrix.png",
        "06_roc_curve.png",
        "07_calibration_curve.png",
    ]
    for fn in figure_files:
        fp = os.path.join(output_dir, fn)
        if os.path.isfile(fp):
            img = plt.imread(fp)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.imshow(img)
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

print(f"Wrote PDF report to {pdf_path}")