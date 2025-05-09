#STUDENT DETAILS -------------------------------------
#NAME : MADDHAV SUNEJA
#N-ID : N14893153
#NetId : ms14565


#####IMPORTS ------------------------------------------

import pandas as pd
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, roc_curve,
                             accuracy_score, precision_score,
                             recall_score, f1_score,classification_report)
from scipy.stats import pearsonr

# MY N-ID SEED -------------------------------------------------
SEED = 14893153
import random
random.seed(SEED)
np.random.seed(SEED)

#####CLEANING THE DATA ------------------------------------------

# Load original CSV
df = pd.read_csv('rmpCapstoneNum.csv')
dp = pd.read_csv('rmpCapstoneQual.csv')

# Drop rows where all values are NaN or 0
df_cleaned = df[~((df.isna() | (df == 0)).all(axis=1))]

# Replace all NaN with a '-1' as a placeholder in 'WouldTakeAgain' Column
df_cleaned.iloc[:, 4] = df_cleaned.iloc[:, 4].fillna(-1)
df_cleaned.columns.values[4] = '-1'

#Save the current header (which are integers)
old_header = df_cleaned.columns.tolist()

# Add the old header as the first row in the DataFrame
df_cleaned.loc[-1] = old_header  # Insert at index -1
df_cleaned.index = df_cleaned.index + 1  # Shift index
df_cleaned = df_cleaned.sort_index().reset_index(drop=True)  # Reorder and reset index

#Set new string-based headers
df_cleaned.columns = ['Avg Rating', 'Avg Difficulty', 'No. of Ratings', 'Pepper', 'WouldTakeAgain',
                      'No. of Ratings from online', 'Male gender', 'Female']

# Ensure all the columns have the correct data type 
df_cleaned = df_cleaned.apply(pd.to_numeric, errors='raise')  
df_cleaned['No. of Ratings from online'] = df_cleaned['No. of Ratings from online'].astype(int)
df_cleaned['Male gender'] = df_cleaned['Male gender'].astype(int)
df_cleaned['Pepper'] = df_cleaned['Pepper'].astype(int)
df_cleaned['No. of Ratings'] = df_cleaned['No. of Ratings'].astype(int)

# remove rows with logically impossible gender (Male==1 & Female==1)
df_cleaned = df_cleaned[~((df_cleaned["Male gender"] == 1) & (df_cleaned["Female"] == 1))]

# threshold reliability filter: keep professors with ≥ 3 ratings
df_cleaned = df_cleaned[df_cleaned["No. of Ratings"] >= 3].copy()

# reset the index after all the change 
df_cleaned.reset_index(drop=True, inplace=True)

pd.set_option('display.max_columns', None)
print(df_cleaned.head())

#---------------------------------------------------------------
# Q1) Activists have asserted that there is a strong gender bias in student evaluations of professors, with
# male professors enjoying a boost in rating from this bias. While this has been celebrated by ideologues,
# skeptics have pointed out that this research is of technically poor quality, either due to a low sample
# size – as small as n = 1 (Mitchell & Martin, 2018), failure to control for confounders such as teaching
# experience (Centra & Gaubatz, 2000) or obvious p-hacking (MacNell et al., 2015). We would like you to
# answer the question whether there is evidence of a pro-male gender bias in this dataset.
# Hint: A significance test is probably required.

df_q1 = df_cleaned[((df_cleaned["Male gender"] == 1) & (df_cleaned["Female"] == 0)) |
                   ((df_cleaned["Male gender"] == 0) & (df_cleaned["Female"] == 1))].copy()

# 1️⃣ Welch's t-test (two-sample test)
male_ratings = df_q1[df_q1["Male gender"] == 1]["Avg Rating"]
female_ratings = df_q1[df_q1["Female"] == 1]["Avg Rating"]

t_stat, p_val = ttest_ind(male_ratings, female_ratings, equal_var=False)
p_one_tailed = p_val / 2 if t_stat > 0 else 1 - (p_val / 2)

print(f"Welch's t-test:\n t = {t_stat:.4f}, one-tailed p = {p_one_tailed:.6f}")
print("Significant at α = 0.005?", p_one_tailed < 0.005)

# 2️⃣ OLS Regression: Avg Rating ~ Male gender + No. of Ratings
X = df_q1[["Male gender", "No. of Ratings"]]
X = add_constant(X)
y = df_q1["Avg Rating"]

model = sm.OLS(y, X).fit()
print("\nOLS Regression Results (Controlling for Experience):")
print(model.summary(alpha=0.005))


#---------------------------------------------------------------
# Q2) Is there an effect of experience on the quality of teaching? You can operationalize quality with
# average rating and use the number of ratings as an imperfect – but available – proxy for experience.
# Again, a significance test is probably a good idea.

X_q2 = add_constant(df_cleaned["No. of Ratings"])
y_q2 = df_cleaned["Avg Rating"]

model_q2 = sm.OLS(y_q2, X_q2).fit()
print(model_q2.summary())

plt.figure(figsize=(8, 6))
sns.regplot(x="No. of Ratings", y="Avg Rating", data=df_cleaned, scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
plt.title("Q2: Relationship Between Experience and Avg Rating")
plt.xlabel("Number of Ratings (Experience Proxy)")
plt.ylabel("Average Rating")
plt.grid(True)
plt.tight_layout()
plt.show()

#---------------------------------------------------------------
# Q3) What is the relationship between average rating and average difficulty?

x_q3 = df_cleaned["Avg Difficulty"]
y_q3 = df_cleaned["Avg Rating"]

# Pearson correlation
r_q3, p_q3 = pearsonr(x_q3, y_q3)
print(f"Q3 • Pearson r = {r_q3:.3f}, p = {p_q3:.3e}, Significant @ alpha =0.005? {p_q3 < 0.005}")

# OLS Regression
X_q3 = add_constant(x_q3)
model_q3 = sm.OLS(y_q3, X_q3).fit()
print(model_q3.summary())

# Plot
sns.regplot(x=x_q3, y=y_q3, line_kws={"color": "red"})
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title("Q3: Average Rating vs. Difficulty")
plt.show()


#---------------------------------------------------------------
# Q4 Do professors who teach a lot of classes in the online modality receive higher or lower ratings than
# those who don’t? Hint: A significance test might be a good idea, but you need to think of a creative but
# suitable way to split the data

df = df_cleaned.copy()
df["online_share"] = (
    df["No. of Ratings from online"] / df["No. of Ratings"]
).clip(0, 1)

# 2.  Split at the 75-th percentile  →  “mostly-online” vs “mostly-in-person”
q50 = df["online_share"].quantile(0.50)          # creative, data-driven cut-off
df["online_group"] = np.where(df["online_share"] >= q50,
                              "mostly_online", "mostly_inperson")

# 3.  Welch’s two-tailed t-test   (α = 0.005)
from scipy.stats import ttest_ind
online     = df[df["online_group"]=="mostly_online"]["Avg Rating"]
inperson   = df[df["online_group"]=="mostly_inperson"]["Avg Rating"]

t_stat, p_two = ttest_ind(online, inperson, equal_var=False)
alpha = 0.005
print(f"t = {t_stat:.3f},  p(two-tailed) = {p_two:.4e}")
print("Significant at α = 0.005?", p_two < alpha)

# 4.  Effect size  (Cohen’s d)
n1, n2   = len(online), len(inperson)
pooled_sd = np.sqrt(((n1-1)*online.var(ddof=1) + (n2-1)*inperson.var(ddof=1)) / (n1+n2-2))
d = (online.mean() - inperson.mean()) / pooled_sd
print(f"Cohen’s d = {d:.2f}")

# 5.  Quick visual
sns.boxplot(x="online_group", y="Avg Rating", data=df)
plt.title("Average Rating by Teaching-Modality Share")
plt.xlabel(f"Group (cut at 50-th-pct online_share = {q50:.2f})")
plt.ylabel("Average Rating")
plt.show()

#---------------------------------------------------------------
# Q5) What is the relationship between the average rating and the proportion of people who would take
# the class the professor teaches again?

# 1. Filter valid rows (exclude placeholder -1)
df_q5 = df_cleaned[df_cleaned['WouldTakeAgain'] >= 0].copy()

# 2. Extract variables
x_q5 = df_q5['WouldTakeAgain']
y_q5 = df_q5['Avg Rating']

# 3. Pearson correlation
r_q5, p_q5 = pearsonr(x_q5, y_q5)
print(f"Pearson r = {r_q5:.3f}, p = {p_q5:.3e}, Significant @ α=0.005? {p_q5 < 0.005}")

# 4. OLS Regression: Avg Rating ~ WouldTakeAgain
X_q5 = add_constant(x_q5)
model_q5 = sm.OLS(y_q5, X_q5).fit()
print(model_q5.summary())

# 5. Plot
sns.regplot(x=x_q5, y=y_q5, line_kws={"color": "red"})
plt.xlabel("Proportion of Students Who Would Take Again")
plt.ylabel("Average Rating")
plt.title("Q5: Avg Rating vs. Would-Take-Again Proportion")
plt.show()

#---------------------------------------------------------------
# Q6) Do professors who are “hot” receive higher ratings than those who are not? 
# Again, a significance test is indicated.

# Filter groups
hot_group = df_cleaned[df_cleaned.iloc[:, 3] == 1].iloc[:, 0]
not_hot_group = df_cleaned[df_cleaned.iloc[:, 3] == 0].iloc[:, 0]

# Welch’s t-test (one-tailed)
t_stat, p_val = ttest_ind(hot_group, not_hot_group, equal_var=False)

# Adjust for one-tailed test
p_val_one_tailed = p_val / 2 if t_stat > 0 else 1 - (p_val / 2)

print(p_val_one_tailed)

import seaborn as sns
import matplotlib.pyplot as plt

# Use iloc to extract columns directly by index
sns.boxplot(x=df_cleaned.iloc[:, 3], y=df_cleaned.iloc[:, 0])

# Add labels manually
plt.xlabel("Hotness (0 = Not Hot, 1 = Hot)")
plt.ylabel("Average Rating")
plt.title("Average Rating by 'Hotness' (Using Column Indexes)")

plt.show()

#---------------------------------------------------------------

# Q7) Build a regression model predicting average rating from difficulty (only). 
# Make sure to include the R2 and RMSE of this model.

# Define predictor (X) and target (y)
X = df_cleaned[['Avg Difficulty']]  # 2D input for sklearn
y = df_cleaned['Avg Rating']        # 1D output

# Build and fit the regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Output results
print(f"R² (coefficient of determination): {r2:.4f}")
print(f"RMSE (root mean squared error): {rmse:.4f}")

sns.lmplot(data=df_cleaned, x='Avg Difficulty', y='Avg Rating', line_kws={"color": "red"})
plt.title('Avg Rating vs Avg Difficulty')
plt.show()

#---------------------------------------------------------------

# Q8) Build a regression model predicting average rating from all available factors. Make sure to include
# the R2 and RMSE of this model. Comment on how this model compares to the “difficulty only” model
# and on individual betas. Hint: Make sure to address collinearity concerns.

X = df_cleaned.drop(columns=['Avg Rating', 'WouldTakeAgain'])
y = df_cleaned['Avg Rating']

corr = X.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Predictors")
plt.show()

X = df_cleaned[['Avg Difficulty', 'No. of Ratings', 'Pepper', 'No. of Ratings from online', 'Male gender']]
X = add_constant(X)

y = df_cleaned['Avg Rating']

model = sm.OLS(y, X).fit()
# print summary with 99.5 % CIs  (alpha = .005)
print(model.summary2(alpha=0.005))

# flag which predictors are significant at α = 0.005
print("\nSignificant at alpha = 0.005?\n", model.pvalues < 0.005)

#---------------------------------------------------------------

# Q9) Build a classification model that predicts whether a professor receives a “pepper” from average
# rating only. Make sure to include quality metrics such as AU(RO)C and also address class imbalances.

# 1. Data
X = df_cleaned[['Avg Rating']].values
y = df_cleaned['Pepper'].values.astype(int)

# 2. Train-test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Logistic Regression with class_weight
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# 4. Predictions and Probabilities
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 5. Evaluation Metrics
auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"AUROC   : {auc:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall   : {rec:.3f}")
print(f"F1-score : {f1:.3f}")

# 6. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: Predicting Pepper from Avg Rating')
plt.legend()
plt.show()

#---------------------------------------------------------------

# Q10) Build a classification model that predicts whether a professor receives a “pepper” from all available
# factors. Comment on how this model compares to the “average rating only” model. Make sure to
# include quality metrics such as AU(RO)C and also address class imbalances.

### Part 1 steps to decide which features to use for the classification model -----

# Base AUC for all features
X_all = df_cleaned[["Avg Rating", "Avg Difficulty", "No. of Ratings", 
                    "No. of Ratings from online", "Male gender", "Female"]]
y = df_cleaned["Pepper"]

# Base model AUC
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y, test_size=0.3, random_state=SEED)
base_model = LogisticRegression(max_iter=1000).fit(X_train_all, y_train_all)
base_auc = roc_auc_score(y_test_all, base_model.predict_proba(X_test_all)[:, 1])
print(f"Base AUC with all features: {base_auc:.3f}\n")

# Loop through and drop one feature at a time
for col in X_all.columns:
    X_reduced = X_all.drop(columns=[col])
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=SEED)
    
    temp_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    auc = roc_auc_score(y_test, temp_model.predict_proba(X_test)[:, 1])
    
    print(f"Dropped '{col}': AUC = {auc:.3f} (Δ = {auc - base_auc:+.3f})")

### Building the actual classification model now  ------

# Step 1: Use cleaned data where Pepper is defined
df_model = df_cleaned[df_cleaned["Pepper"].isin([0, 1])].copy()

# Step 2: Define features and target
X = df_model[["Avg Rating", "Avg Difficulty", "No. of Ratings"]]
y = df_model["Pepper"]

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Step 4: Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Evaluate performance
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC (All Predictors): {auc:.3f}")

# Step 6: Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"All Predictors (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Pepper Prediction (All Predictors)")
plt.legend()
plt.grid(True)
plt.show()