# DS_CAPSTONE
Capstone project for DS UA 112 ;)


# 📊 Assessing Professor Effectiveness (APE)

> Capstone project for *Principles of Data Science – DS-UA 112*, NYU

This project investigates bias and rating patterns in student evaluations of professors using a dataset of 89,000+ entries from [RateMyProfessors.com](https://www.ratemyprofessors.com/). It addresses questions around gender bias, course difficulty, teaching modality, and "hotness" indicators, and builds predictive models to analyze and forecast student rating behavior.

---

## 🔍 Project Questions Explored

1. Is there evidence of gender bias in student evaluations?
2. Does teaching experience correlate with higher ratings?
3. How does course difficulty affect perceived teaching quality?
4. Do online instructors receive different ratings than in-person ones?
5. Does the willingness to retake a class predict quality ratings?
6. Do “hot” professors get higher ratings?
7. How well can we model rating based on difficulty alone?
8. How well can we model rating using all available features?
9. Can we predict “pepper” status (hotness) using average rating?
10. Can we build a better “pepper” prediction model using all features?

---

## 🛠️ Tools & Technologies

- **Python** (v3.10)
- **pandas**, **numpy**
- **matplotlib**, **seaborn**
- **scikit-learn** (logistic regression, train-test split, metrics)
- **statsmodels** (OLS regression)
- Significance testing (Welch’s t-test, Pearson correlation)
- AUROC, Precision, Recall, F1, R², RMSE

---

## 📈 Sample Insights

- 📉 **Difficulty vs Rating**: Negative correlation (r = -0.59), R² = 0.35
- 👨‍🏫 **Gender Bias**: Male professors rated higher (t = 5.56, p < 0.000001)
- 🔥 **“Hotness” Effect**: Statistically significant rating boost for "pepper" professors
- 🔁 **Would Take Again**: Strongest predictor of high ratings (R² = 0.776)
- 🤖 **Model Performance**:
  - Pepper prediction using Avg Rating: AUROC = 0.747
  - Full-featured model: AUROC = 0.774

---

## 🧼 Preprocessing

- Removed rows with all-zero or all-null values
- Filtered out professors with < 3 ratings
- Corrected gender inconsistencies (e.g., Male = 1 and Female = 1)
- Handled missing `WouldTakeAgain` using -1 placeholder
- Seeded RNG using last 8 digits of student ID: `SEED = 14893153`

---

## 📁 Project Structure

```plaintext
APE-capstone/
├── MS_DS.pdf            # Final report with analysis, visualizations, and interpretations
├── ape_capstone.py      # Full Python script for data cleaning, analysis, and modeling
├── README.md            # Project description and structure (this file)