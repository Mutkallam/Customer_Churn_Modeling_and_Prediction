# Customer Churn Modeling and Prediction
## Tools: Python (Pandas, NumPy), Scikit-learn, Logistic Regression, Gradient Boosting, GridSearchCV, Matplotlib/Seaborn

In this project, I developed and evaluated machine learning models to predict customer churn using the Telco Customer Churn dataset and translated model outputs into actionable retention strategies through risk segmentation.

## Key Steps:
Dataset Preparation: Cleaned and prepared the telecom dataset by converting TotalCharges to numeric, handling missing values, dropping customer identifiers, encoding categorical variables, and engineering interpretable features such as new_customer, high_monthly_charge, multiple_support, and streaming_bundle. Performed a stratified 80/20 train-test split to preserve the original churn distribution.

### Exploratory Data Analysis:
Analyzed churn behavior across contract types, tenure ranges, monthly charges, payment methods, and service add-ons to identify key risk drivers. Examined how early tenure, flexible contracts, and pricing intensity relate to higher churn probability, and used these insights to guide feature engineering.

### Feature Engineering:
Created nonlinear and engagement-based features to capture business-relevant behaviors, including early-stage customer risk (tenure ≤ 12 months), high pricing thresholds (top 25% of monthly charges), bundled support engagement (count of security and support services), and streaming bundle usage.

### Modeling Approach:
Built a Logistic Regression baseline model using standardized numeric features and one-hot encoded categorical variables, then optimized the classification threshold to 0.4 to prioritize recall due to the higher cost of missed churners. 
Trained a Gradient Boosting classifier and performed 5-fold cross-validated hyperparameter tuning using GridSearchCV to improve ranking performance and generalization.

### Model Performance:
The tuned Gradient Boosting model achieved a cross-validated ROC-AUC of 0.8436 and a test ROC-AUC of 0.8622. At a business-aligned threshold of 0.4, the model achieved 65% precision and 66% recall for churners, correctly identifying 258 out of 374 actual churners while maintaining overall accuracy of 81%. 
The Gradient Boosting model outperformed the logistic regression baseline (ROC-AUC ≈ 0.857) and provided a stronger balance between precision and recall.

### Risk Segmentation:
Customers were segmented into three risk tiers based on predicted churn probability. The High Risk segment (152 customers) exhibited a realized churn rate of 73.7%, the Medium Risk segment (226 customers) churned at 58.8%, and the Low Risk segment (1,031 customers) churned at only 12.5%. This strong separation demonstrates strong ranking performance and practical deployment value.

### Key Takeaways:
Contract structure, tenure length, pricing intensity, and service engagement significantly influence churn risk. 
Fiber optic service, electronic check payments, streaming add-ons, and short tenure increased churn likelihood, while long-term contracts, extended tenure, bundled services, and support features reduced risk. By combining predictive modeling with risk segmentation, this project demonstrates how machine learning can guide targeted retention strategies, reduce unnecessary incentive spending, and improve customer retention outcomes.
