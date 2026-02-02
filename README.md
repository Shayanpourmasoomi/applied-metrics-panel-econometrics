# Applied Metrics – Problem Set 05  
## Panel Data Models and Longitudinal Analysis

This repository contains the solution to **Problem Set 05** of the *Applied Metrics (MSc)* course (Spring 2025).  
The project applies a wide range of **panel data estimators** to study labor supply, wages, and health outcomes using classic longitudinal datasets.

---

## Files

- **applied-metrics-panel-econometrics.md**  
  Main analysis file containing all empirical results, tables, and interpretations.

- **PS_05.pdf**  
  Official problem set description.

---

## Datasets

This project uses three well-known panel datasets:

### 1. MOM (Wage–Hours Panel)
- Variables: log hours, log wages, children, age, disability, individual ID, year
- Used to estimate labor supply elasticity
- 50% random subsample is drawn for estimation

### 2. WAGEPAN
- Individual-level wage panel
- Variables include wages, education, race, ethnicity, and year dummies
- Used to compare pooled OLS, FE, RE, and FD estimators

### 3. LOWBIRTH
- State-level panel data (1987–1990)
- Outcome: percentage of low birth weight births
- Key regressor: AFDC participation rate
- Controls: income, population, physicians, hospital beds

---

## Methods and Estimators

The following panel data estimators are implemented and compared:

- Pooled OLS
- Between Estimator
- Fixed Effects (Within)
- First Differences (FD)
- Random Effects (GLS)
- Random Effects (MLE)
- Panel Bootstrap (clustered at individual level)
- Robust and serial-correlation–robust standard errors

Libraries used include:
- `statsmodels`
- `linearmodels`
- `pandas`, `numpy`

---

## Analysis Overview

### Question 1 – Labor Supply Elasticity
- Estimation of the effect of wages on hours worked
- Comparison across six estimators
- Panel bootstrap standard errors with 200 replications
- Discussion of:
  - Consistency
  - Bias
  - Interpretation of labor supply elasticity
  - Differences between default and bootstrap standard errors

---

### Question 2 – Wage Dynamics (WAGEPAN)
- Estimation with year dummies using:
  - Pooled OLS
  - Random Effects
  - Fixed Effects
  - First Differences
- Inclusion of time-invariant covariates (education, race, ethnicity)
- Comparison of:
  - Coefficients across estimators
  - Ordinary vs fully robust standard errors
- Discussion of reliability of standard error estimators

---

### Question 3 – Low Birth Weight and Welfare Policy (LOWBIRTH)
- Estimation of pooled OLS with usual and HAC standard errors
- First-difference estimation to remove state fixed effects
- Interpretation of AFDC participation effects
- Extension to a quadratic specification
- Calculation and interpretation of the turning point

---

## Key Econometric Concepts

- Unobserved heterogeneity
- Fixed vs random effects
- Serial correlation
- Robust and clustered standard errors
- Panel bootstrap
- Functional form misspecification
- Causal interpretation in panel settings

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
