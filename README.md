# Customer Churn Prediction (Logistic Regression)

## 1. Business Problem

Customer churn directly impacts revenue and long-term growth.  
The goal is to:

- Identify customers at **high risk of churn**
- Prioritize retention interventions
- Optimize outreach cost vs revenue saved

Churn is expensive because acquiring a new customer costs more than retaining one.

---

## 2. Business Objectives

Depending on strategy, we can optimize:

### A) Maximize Recall (Aggressive Retention Strategy)
Catch as many churners as possible.
- Useful when retention offers are low-cost.
- Accepts more false positives.

### B) Maximize Precision (Selective Outreach)
Target only customers very likely to churn.
- Useful when retention campaigns are expensive.

### C) Cost-Based Optimization (Recommended)
Minimize:

Expected Cost = (Cost_FP × False Positives) + (Cost_FN × False Negatives)

This is implemented via automatic threshold optimization.

---

## 3. Customers to Target

After prediction:

High-risk customers = probability >= optimized threshold

You can:
- Rank customers by churn probability
- Target top 10–20%
- Segment by contract type, tenure, or monthly charges

---

## 4. Retention Policies to Suggest

Based on typical churn drivers:

1. Month-to-month contract customers → Offer discounted long-term contract.
2. High monthly charges → Offer bundled discount.
3. No online security/tech support → Offer free add-on trial.
4. Short tenure customers → Early loyalty rewards.
5. Electronic check payment users → Incentivize autopay setup.

Policies should align with coefficient insights from logistic regression.

---

## 5. Model Approach

We use:

- Logistic Regression (interpretable baseline)
- Feature engineering with ColumnTransformer
- Cross-validation
- ROC-AUC, Precision, Recall, F1 evaluation
- Threshold optimization module

---

## 6. Should We Use Only Logistic Regression?

Logistic Regression is:

✔ Interpretable  
✔ Fast  
✔ Strong baseline  
✔ Easy to deploy  

However, in production you should compare with:

- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- CatBoost

Tree models often achieve higher ROC-AUC but are less interpretable.

Best practice:
1. Start with logistic regression (baseline).
2. Compare against tree-based models.
3. Use SHAP if interpretability is required.

---

## 7. Threshold Optimization

Run:

python -m src.train --optimize f1  
python -m src.train --optimize recall  
python -m src.train --optimize cost  

This automatically selects the best probability cutoff.

---

## 8. Final Deliverables

- Trained model pipeline
- CV metrics
- ROC & PR curves
- Confusion matrix
- Top churn drivers
- Business recommendations

---

This project is structured to resemble a production ML workflow suitable for interviews, portfolios, and real-world applications.
