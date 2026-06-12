# Credit Card Fraud Detection Dataset - Personal Notes

Dataset Name: Credit Card Fraud Detection Dataset

Purpose:
I am using this dataset for my Big Data project to perform fraud detection using PySpark and machine learning techniques. The objective is to classify whether a credit card transaction is legitimate or fraudulent.

Dataset Details:

* Total Transactions: 284,807
* Total Features: 31 columns
* Legitimate Transactions: 284,315
* Fraudulent Transactions: 492

Column Description:

1. Time

* Represents the number of seconds elapsed between the current transaction and the first transaction in the dataset.
* It is not the actual date or clock time.

2. V1 to V28

* These are anonymized features generated using Principal Component Analysis (PCA).
* The original transaction details were hidden to protect customer privacy.
* Their exact meanings are unknown, but they contain patterns useful for fraud detection.

3. Amount

* Indicates the monetary value of the transaction.
* Can help identify unusual transaction behaviour.

4. Class (Target Variable)

* 0 = Legitimate Transaction
* 1 = Fraudulent Transaction

Important Observation:
The dataset is highly imbalanced.

* Legitimate transactions account for approximately 99.83%.
* Fraudulent transactions account for approximately 0.17%.

This means that accuracy alone is not a reliable evaluation metric. Metrics such as Precision, Recall, F1-Score, and ROC-AUC should also be considered.

Problem Statement:
Using the transaction features (Time, V1-V28, and Amount), build a model that can accurately identify fraudulent credit card transactions.

Reminder to Myself:

* Don't panic when you see V1 to V28. Nobody knows what they originally meant.
* Check for class imbalance before training models.
* Avoid relying only on accuracy.
* Document every preprocessing step and save the code properly.
* Future me: if you're reading this at 2 AM before submission, breathe. You already understood this dataset once. You can do it again.

Project Goal:
Detect the 492 fraudulent transactions hidden among 284,315 normal transactions as efficiently and accurately as possible.
