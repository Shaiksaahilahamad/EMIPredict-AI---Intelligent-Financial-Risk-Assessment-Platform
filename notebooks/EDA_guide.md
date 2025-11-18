# EDA Guide for EMIPredict AI

Suggested analyses:

1. **Target distributions**
   - `emi_eligibility` value counts by scenario
   - Distribution of `max_monthly_emi`

2. **Correlation & relationships**
   - Correlation heatmap of numeric features
   - Boxplots of `max_monthly_emi` vs `emi_scenario`
   - Credit score vs EMI eligibility

3. **Risk & affordability**
   - Distribution of `dti_ratio`, `expense_to_income_ratio`
   - Compare these ratios by `emi_eligibility` class

4. **Demographic patterns**
   - Age vs eligibility
   - Education vs eligibility
   - Employment type vs eligibility

Use this as a checklist for your EDA notebook.
