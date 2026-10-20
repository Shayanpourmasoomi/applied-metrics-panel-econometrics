# Panel Data Econometrics: Labour Supply, Wages and Infant Health

Estimation and comparison of the main **panel-data estimators** on three classic longitudinal datasets. The project asks how much each estimator's answer depends on its assumptions about unobserved heterogeneity.

## Applications

| Dataset | Question |
|---|---|
| **Wage–hours panel (MOM)** | Intertemporal labour-supply elasticity: how hours respond to wages |
| **WAGEPAN** (Vella and Verbeek) | Returns to education and wage dynamics under different assumptions about unobserved ability |
| **LOWBIRTH** (US states, 1987–1990) | Whether welfare (AFDC) participation is associated with low birth weight, net of state effects |

## Estimators

- Pooled OLS and the between estimator
- Fixed effects (within) and first differences
- Random effects (GLS and maximum likelihood)
- Cluster-robust and serial-correlation-robust standard errors
- Panel bootstrap, clustered at the individual level

The write-up compares the estimators side by side and interprets their differences in terms of correlation between regressors and unobserved individual effects.

## Repository structure

```
analysis.md   # full analysis: code, estimation tables and interpretation
```

Figures and tables are generated when the code is run; the figure files themselves are not stored in the repository.

## Reproducing

```bash
pip install pandas numpy statsmodels linearmodels
```

WAGEPAN and LOWBIRTH are available in the Wooldridge datasets (e.g. the [`wooldridge`](https://pypi.org/project/wooldridge/) Python package).

## Context

Developed during the MSc in Economics at Sharif University of Technology (Applied Econometrics, Spring 2025).

---
**Shayan Pourmasoomi** · MSc Economics, Sharif University of Technology · [LinkedIn](https://www.linkedin.com/in/shayan-pourmasoomi-8b6206389)
