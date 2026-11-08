# Online shopper purchase prediction

## a. Problem statement

An online store records every visit as a session: which kinds of pages were opened, how long the visitor stayed, how often those pages are left immediately, and a few facts about the visit itself (month, browser, traffic source, new versus returning visitor, weekend). The practical question is whether that session ends in a purchase.

This is a binary classification problem. The positive class is a session with `Revenue = 1`. About 15.47% of sessions buy something (1,908 of 12,330), so a classifier that always predicts "no purchase" already scores about 84.5% accuracy. Accuracy alone is therefore a weak way to judge the models. Precision, recall, and F1 are reported for the purchase class, together with ROC AUC and the Matthews correlation coefficient, which uses all four cells of the confusion matrix.

The fitted models are meant to score a fresh session without being retrained. A held-out test file is what the Streamlit app evaluates.

## b. Dataset description

**Source.** Online Shoppers Purchasing Intention Dataset, UCI Machine Learning Repository.  
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The sessions were analyzed by Sakar, Polat, Katircioglu, and Kastro in *Neural Computing and Applications* (2019), "Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks."

**Size.** 12,330 sessions and 17 input features. That meets the assignment minimum of 12 features and 500 instances. The file has no missing values. The copy used for training is `data/online_shoppers_intention.csv`.

**Target.** `Revenue`: 1 if the session ended in a transaction, 0 otherwise. Positive class for precision, recall, and F1: purchase (`1`).

**Class balance.** 1,908 purchases and 10,422 sessions with no purchase.

**Features.**

| Feature | Meaning |
| --- | --- |
| Administrative | Count of administrative pages viewed |
| Administrative_Duration | Time spent on administrative pages |
| Informational | Count of informational pages viewed |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Count of product-related pages viewed |
| ProductRelated_Duration | Time spent on product-related pages |
| BounceRates | Average bounce rate of the pages visited |
| ExitRates | Average exit rate of the pages visited |
| PageValues | Average page value of the pages visited |
| SpecialDay | Closeness of the visit to a special day, from 0 to 1 |
| Month | Month of the visit |
| OperatingSystems | Operating-system code |
| Browser | Browser code |
| Region | Geographic region code |
| TrafficType | Traffic-source code |
| VisitorType | New visitor, returning visitor, or other |
| Weekend | Whether the visit fell on a weekend |

Page counts, durations, rates, page value, and special-day closeness are numeric. Month, operating system, browser, region, traffic type, visitor type, and weekend are categories. The integer codes are not magnitudes, so they are one-hot encoded. Unseen categories in an uploaded file are ignored. Numeric columns are standardized only inside the logistic regression, k-nearest neighbor, and linear SVM pipelines.

**Split used for the table below.** Stratified 80/20 split, `random_state = 42`. The training portion has 9,864 sessions. The test portion has 2,466 sessions and is stored as `test_data.csv` with the original 17 features plus `Revenue`. Hyperparameters were chosen by 3-fold stratified cross-validation on the training portion only, scoring purchase-class F1. The test file was scored once, after the pipelines were refit on the full training portion. The app loads those saved pipelines and does not train again.

## c. Github Repository Link

Replace this line with the public repository URL after you push this folder:

`https://github.com/<your-username>/<your-repo-name>`

The repository root contains:

- `app.py` — Streamlit evaluation app
- `train_models.py` — training and the held-out split
- `preprocessing.py`, `evaluation.py`, `model_catalog.py` — shared by training and the app
- `requirements.txt`
- `README.md`
- `test_data.csv` — held-out sessions, including `Revenue`
- `model/` — five saved pipelines (`.joblib`) and `training_notes.json`
- `data/online_shoppers_intention.csv` — full UCI file

## d. Models used

All five models are scikit-learn classifiers trained on the same sessions and the same 17 features.

| Model | Setting chosen on the training folds |
| --- | --- |
| Logistic Regression | `C = 0.1`, `class_weight = balanced`, L2, lbfgs |
| Decision Tree | `max_depth = 6`, `min_samples_leaf = 10`, `class_weight = balanced` |
| KNN | `n_neighbors = 5`, `weights = distance`, features standardized |
| Naive Bayes | Multinomial NB, `alpha = 5.0`, features left non-negative (no scaling) |
| Support Vector Machine | Linear SVM (`LinearSVC`), `C = 0.1`, `class_weight = balanced` |

The SVM is linear so the saved model is a weight vector. A kernel SVM would have stored a large set of support vectors and is a poor fit for the Streamlit Community Cloud size limits. AUC for the linear SVM uses the signed decision value of the purchase class. The other four models use the predicted probability of a purchase.

Multinomial Naive Bayes is the Naive Bayes variant in this project. Page counts, durations, and rates are non-negative, and the context fields become 0/1 columns. A Gaussian Naive Bayes model on the same one-hot matrix labeled about 83% of test sessions as purchases and dropped to about 0.31 accuracy, so it was not kept.

### Comparison on the held-out test file (2,466 sessions)

Precision, recall, and F1 are for the purchase class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8451 | 0.8949 | 0.5000 | 0.7382 | 0.5962 | 0.5194 |
| Decision Tree | 0.8491 | 0.9096 | 0.5080 | 0.8325 | 0.6310 | 0.5692 |
| KNN | 0.8747 | 0.7655 | 0.6682 | 0.3796 | 0.4841 | 0.4407 |
| Naive Bayes | 0.8751 | 0.7180 | 0.5995 | 0.5838 | 0.5915 | 0.5179 |
| Support Vector Machine | 0.8402 | 0.8890 | 0.4895 | 0.7304 | 0.5861 | 0.5070 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Balanced logistic regression is the most even linear model. AUC is 0.8949 and MCC is 0.5194. It recovers 73.82% of purchases. Precision is 0.50, so half of the sessions it flags do not buy. The largest weight is page value (positive). Exit rate has a clear negative weight: sessions that reach high-exit pages are less likely to purchase. November increases the log-odds; February, March, and May decrease them. |
| Decision Tree | Best ranking and best purchase detection together: AUC 0.9096, recall 0.8325, F1 0.6310, MCC 0.5692. Page value accounts for about 81% of the split importance, which matches the logistic result and the shopping meaning of that feature (pages the visitor saw already tend to lead to a transaction). November is the next split. Depth 6 and a leaf size of 10 stop the tree from cutting on rare browser and traffic codes. Accuracy stays near the logistic model because the balanced class weights spend some correct majority predictions in order to catch more buyers. |
| KNN | Highest accuracy of the distance-based models (0.8747) comes with the weakest purchase recall (0.3796) and a mid-range AUC (0.7655). With a 15% purchase rate, the five nearest standardized sessions are often non-buyers, so the vote misses most real purchases. Precision on the purchases it does predict is the best of the five (0.6682). KNN has no class weighting, and that shows up directly in the gap between accuracy and recall. |
| Naive Bayes | Multinomial NB with `alpha = 5` is well calibrated to the base rate: precision 0.5995 and recall 0.5838, accuracy 0.8751, MCC 0.5179. F1 is close to logistic regression. AUC is the lowest of the five (0.7180), so the probability is a weaker ranking signal than the tree or the linear models. The independence assumption is strained because bounce rate, exit rate, and page value move together, and because a long product-page duration dominates the raw counts. Smoothing at 5.0 was preferred on the training folds and keeps rare browser and traffic codes from dominating. |
| Support Vector Machine | The linear SVM tracks logistic regression on every metric (AUC 0.8890, F1 0.5861, MCC 0.5070, recall 0.7304). Page value is again the strongest positive measurement. A few sparse browser and traffic codes also receive large weights, which is a reason the margin model is a step behind the tree: those codes describe few sessions. The linear decision is still strong enough to rank sessions (AUC near 0.89) and is cheap to deploy. |
| Overall, Winner for your dataset? | **Decision Tree.** It leads on AUC, purchase recall, purchase F1, and MCC at the same time. KNN and Multinomial NB post slightly higher accuracy, but both are weaker at ranking sessions and, in the case of KNN, miss most purchases. On a 15% purchase rate, MCC and purchase F1 are the fair summary, and the depth-limited tree wins both. |

## Run and deploy

Train and write the model files (already done in this folder; run again only if you change the split or the grids):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

On Streamlit Community Cloud, set the Python version to **3.12** so these scikit-learn 1.9.1 pipelines load. Main file: `app.py`. The app opens on `test_data.csv`. The sidebar can replace that file with another CSV that has the same 17 features and a `Revenue` column. The comparison table, the six metrics, the confusion matrix, and the classification report are computed from the saved pipelines only.

