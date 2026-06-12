# 1 After eda n fraud
Fraudulent transactions constituted only 0.1727% of the total transactions. Although fraudulent transactions had a higher average transaction amount ($122.21) compared to legitimate transactions ($88.29), the maximum legitimate transaction amount ($25,691.16) was significantly higher than the maximum fraudulent transaction amount ($2,125.87). This indicates that transaction amount alone cannot reliably distinguish fraudulent activities, highlighting the need for more sophisticated analytical approaches.


# 2 After time analysis 

Time-based analysis revealed that fraudulent transactions were not evenly distributed across the observation period. Although the average expected fraud occurrence was approximately 10 transactions per hour, certain hours exhibited significantly higher fraud counts, with Hour 11 recording 43 frauds and Hour 26 recording 36 frauds. This indicates the presence of temporal patterns and suggests that fraudulent activities may occur in concentrated bursts rather than randomly. Such insights can assist financial institutions in implementing time-sensitive monitoring mechanisms.

# 3 After viz

## Graph 1: Transaction Class Distribution
Shows that fraud transactions are extremely rare.
Only 492 out of 284,807 transactions are frauds (0.17%).
Indicates a highly imbalanced dataset.
Justifies the need for machine learning and metrics beyond accuracy.

## Graph 2: Average Transaction Amount by Class
Fraud transactions have a higher average amount ($122.21) than legitimate ones ($88.29).
However, legitimate transactions had the highest maximum amount.
Concludes that amount alone cannot identify fraud.
Multiple features are needed for prediction.

## Graph 3: Fraud Transactions by Hour
Frauds are not evenly distributed over time.
Certain hours show significantly higher fraud activity.
Suggests the existence of behavioural patterns.
Indicates that predictive models can learn these hidden trends.

## Overall Takeaway
Fraud is rare.
Single features are insufficient.
Fraud exhibits patterns.
Therefore, machine learning is required to detect fraud effectively.

# 4 After heatmap

## Correlation Heatmap

* The heatmap was used to identify features related to fraudulent transactions.
* Features V17 (-0.33), V14 (-0.30), V12 (-0.26), V10 (-0.22), and V16 (-0.20) showed stronger correlations with the target variable (Class).
* Amount had almost no correlation with fraud (0.01), confirming that transaction amount alone cannot detect fraudulent activity.
* These findings indicated that fraud detection depends on the combined effect of multiple features.
* Therefore, machine learning models were required to learn these complex relationships.


## Why Machine Learning?

The exploratory analysis showed that fraudulent transactions are extremely rare, transaction amount alone cannot distinguish fraud, and fraud patterns depend on multiple features and behavioural trends. Therefore, machine learning was required to learn these complex relationships and accurately identify fraudulent transactions.

## Machine Learning Technique Used

For this project, Logistic Regression using Spark MLlib was employed for fraud detection. Logistic Regression is a widely used classification algorithm that predicts whether a transaction is legitimate or fraudulent based on the combined influence of multiple features. It is efficient, interpretable, and well-suited as a baseline model for binary classification problems such as credit card fraud detection.
