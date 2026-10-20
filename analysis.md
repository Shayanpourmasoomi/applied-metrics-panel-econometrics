```python
pip install pandas numpy statsmodels linearmodels
```

    Collecting linearmodels
      Downloading linearmodels-6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.9 kB)
    Collecting mypy-extensions>=0.4 (from linearmodels)
      Downloading mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
    Collecting pyhdfe>=0.1 (from linearmodels)
      Downloading pyhdfe-0.2.0-py3-none-any.whl.metadata (4.0 kB)
    Collecting formulaic>=1.0.0 (from linearmodels)
      Downloading formulaic-1.1.1-py3-none-any.whl.metadata (6.9 kB)
    Collecting setuptools-scm<9.0.0,>=8.0.0 (from setuptools-scm[toml]<9.0.0,>=8.0.0->linearmodels)
      Downloading setuptools_scm-8.3.1-py3-none-any.whl.metadata (7.0 kB)
    Collecting interface-meta>=1.2.0 (from formulaic>=1.0.0->linearmodels)
      Downloading interface_meta-1.3.0-py3-none-any.whl.metadata (6.7 kB)
    Downloading linearmodels-6.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.7/1.7 MB[0m [31m19.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading formulaic-1.1.1-py3-none-any.whl (115 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.7/115.7 kB[0m [31m12.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading mypy_extensions-1.1.0-py3-none-any.whl (5.0 kB)
    Downloading pyhdfe-0.2.0-py3-none-any.whl (19 kB)
    Downloading setuptools_scm-8.3.1-py3-none-any.whl (43 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.9/43.9 kB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading interface_meta-1.3.0-py3-none-any.whl (14 kB)
    Installing collected packages: setuptools-scm, mypy-extensions, interface-meta, pyhdfe, formulaic, linearmodels
    Successfully installed formulaic-1.1.1 interface-meta-1.3.0 linearmodels-6.1 mypy-extensions-1.1.0 pyhdfe-0.2.0 setuptools-scm-8.3.1


```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster
from tqdm import tqdm
```


```python
data = pd.read_stata('lowbirth.dta')


print("first rows:")
print(data.head())


print("\n coulmns:")
print(data.columns)


print("\n Missing values:")
print(data.isnull().sum())


data['d90'] = (data['year'] == 1990).astype(int)


data = data.set_index(['state', 'year'])

print("\n Final data:")
print(data[['lowbrth', 'afdcprc', 'lphypc', 'lbedspc', 'lpcinc', 'lpopul', 'd90']].head())
```

    first rows:
       year  lowbrth  infmort  afdcprt  popul  pcinc  physic   afdcprc  d90  \
    0  1987      8.0     12.2      132   4084  12039     151  3.232125    0   
    1  1990      8.4     10.8      132   4041  14899     158  3.266518    1   
    2  1987      4.8     10.4       19    524  18461     138  3.625954    0   
    3  1990      4.8     10.5       24    550  20867     146  4.363636    1   
    4  1987      6.4      9.5       91   3400  14322     191  2.676471    0   
    
         lpcinc  ...  clbedspc    povrate  cpovrate    afdcpsq  cafdcpsq  \
    0  9.395906  ...       NaN  21.299999       NaN  10.446634       NaN   
    1  9.609050  ... -0.018767  19.200001 -2.099998  10.670140  0.223506   
    2  9.823416  ...       NaN  12.000000       NaN  13.147544       NaN   
    3  9.945924  ... -0.048427  11.400000 -0.600000  19.041323  5.893779   
    4  9.569552  ...       NaN  12.800000       NaN   7.163495       NaN   
    
       physicpc    lphypc   clphypc    lpopul   clpopul  
    0  0.036974 -3.297552       NaN  8.314832       NaN  
    1  0.039099 -3.241652  0.055900  8.304248 -0.010584  
    2  0.263359 -1.334238       NaN  6.261492       NaN  
    3  0.265455 -1.326312  0.007926  6.309918  0.048427  
    4  0.056176 -2.879257       NaN  8.131531       NaN  
    
    [5 rows x 36 columns]
    
     coulmns:
    Index(['year', 'lowbrth', 'infmort', 'afdcprt', 'popul', 'pcinc', 'physic',
           'afdcprc', 'd90', 'lpcinc', 'cafdcprc', 'clpcinc', 'lphysic',
           'clphysic', 'clowbrth', 'cinfmort', 'afdcpay', 'afdcinc', 'lafdcpay',
           'clafdcpy', 'cafdcinc', 'stateabb', 'state', 'beds', 'bedspc',
           'lbedspc', 'clbedspc', 'povrate', 'cpovrate', 'afdcpsq', 'cafdcpsq',
           'physicpc', 'lphypc', 'clphypc', 'lpopul', 'clpopul'],
          dtype='object')
    
     Missing values:
    year         0
    lowbrth      0
    infmort      0
    afdcprt      0
    popul        0
    pcinc        0
    physic       0
    afdcprc      0
    d90          0
    lpcinc       0
    cafdcprc    50
    clpcinc     50
    lphysic      0
    clphysic    50
    clowbrth    50
    cinfmort    50
    afdcpay      0
    afdcinc      0
    lafdcpay     0
    clafdcpy    50
    cafdcinc    50
    stateabb     0
    state        0
    beds         0
    bedspc       0
    lbedspc      0
    clbedspc    50
    povrate      0
    cpovrate    50
    afdcpsq      0
    cafdcpsq    50
    physicpc     0
    lphypc       0
    clphypc     50
    lpopul       0
    clpopul     50
    dtype: int64
    
     Final data:
                  lowbrth   afdcprc    lphypc   lbedspc    lpcinc    lpopul  d90
    state   year                                                                
    alabama 1987      8.0  3.232125 -3.297552 -5.128479  9.395906  8.314832    0
            1990      8.4  3.266518 -3.241652 -5.147247  9.609050  8.304248    1
    alaska  1987      4.8  3.625954 -1.334238 -5.568345  9.823416  6.261492    0
            1990      4.8  4.363636 -1.326312 -5.616771  9.945924  6.309918    1
    arizona 1987      6.4  2.676471 -2.879257 -5.536276  9.569552  8.131531    0


```python

exog_vars = ['afdcprc', 'lphypc', 'lbedspc', 'lpcinc', 'lpopul', 'd90']
X = data[exog_vars]
X = sm.add_constant(X)
y = data['lowbrth']

model_pooled = sm.OLS(y, X).fit()

print("Pooled OLS Results (Usual Standard Errors):")
print(model_pooled.summary())

model_pooled_robust = model_pooled.get_robustcov_results(cov_type='HAC', maxlags=1)
print("\nPooled OLS Results (Serial-Correlation-Robust Standard Errors):")
print(model_pooled_robust.summary())

model_pooled_linearmodels = PooledOLS(y, X).fit()
print("\nPooled OLS Results (linearmodels):")
print(model_pooled_linearmodels.summary)
```

    Pooled OLS Results (Usual Standard Errors):
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                lowbrth   R-squared:                       0.251
    Model:                            OLS   Adj. R-squared:                  0.203
    Method:                 Least Squares   F-statistic:                     5.191
    Date:                Sat, 03 May 2025   Prob (F-statistic):           0.000121
    Time:                        06:02:06   Log-Likelihood:                -142.31
    No. Observations:                 100   AIC:                             298.6
    Df Residuals:                      93   BIC:                             316.9
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         26.5779      7.158      3.713      0.000      12.363      40.792
    afdcprc        0.0956      0.092      1.037      0.302      -0.087       0.279
    lphypc         0.3081      0.715      0.431      0.668      -1.113       1.729
    lbedspc        0.2790      0.513      0.544      0.588      -0.740       1.298
    lpcinc        -2.4947      0.978     -2.550      0.012      -4.437      -0.552
    lpopul         0.7393      0.702      1.053      0.295      -0.655       2.134
    d90            0.5797      0.276      2.099      0.038       0.031       1.128
    ==============================================================================
    Omnibus:                       20.275   Durbin-Watson:                   1.265
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.129
    Skew:                           0.118   Prob(JB):                       0.0769
    Kurtosis:                       1.916   Cond. No.                     1.01e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.01e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    Pooled OLS Results (Serial-Correlation-Robust Standard Errors):
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                lowbrth   R-squared:                       0.251
    Model:                            OLS   Adj. R-squared:                  0.203
    Method:                 Least Squares   F-statistic:                     3.874
    Date:                Sat, 03 May 2025   Prob (F-statistic):            0.00170
    Time:                        06:02:06   Log-Likelihood:                -142.31
    No. Observations:                 100   AIC:                             298.6
    Df Residuals:                      93   BIC:                             316.9
    Df Model:                           6                                         
    Covariance Type:                  HAC                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         26.5779      7.753      3.428      0.001      11.182      41.973
    afdcprc        0.0956      0.103      0.931      0.354      -0.108       0.299
    lphypc         0.3081      0.757      0.407      0.685      -1.195       1.811
    lbedspc        0.2790      0.659      0.424      0.673      -1.029       1.587
    lpcinc        -2.4947      1.015     -2.457      0.016      -4.511      -0.478
    lpopul         0.7393      0.759      0.974      0.333      -0.768       2.247
    d90            0.5797      0.247      2.348      0.021       0.089       1.070
    ==============================================================================
    Omnibus:                       20.275   Durbin-Watson:                   1.265
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):                5.129
    Skew:                           0.118   Prob(JB):                       0.0769
    Kurtosis:                       1.916   Cond. No.                     1.01e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity and autocorrelation robust (HAC) using 1 lags and without small sample correction
    [2] The condition number is large, 1.01e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    
    Pooled OLS Results (linearmodels):
                              PooledOLS Estimation Summary                          
    ================================================================================
    Dep. Variable:                lowbrth   R-squared:                        0.2509
    Estimator:                  PooledOLS   R-squared (Between):              0.2553
    No. Observations:                 100   R-squared (Within):               0.0108
    Date:                Sat, May 03 2025   R-squared (Overall):              0.2509
    Time:                        06:02:06   Log-likelihood                   -142.31
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      5.1912
    Entities:                          50   P-value                           0.0001
    Avg Obs:                       2.0000   Distribution:                    F(6,93)
    Min Obs:                       2.0000                                           
    Max Obs:                       2.0000   F-statistic (robust):             5.1912
                                            P-value                           0.0001
    Time periods:                       2   Distribution:                    F(6,93)
    Avg Obs:                       50.000                                           
    Min Obs:                       50.000                                           
    Max Obs:                       50.000                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          26.578     7.1580     3.7130     0.0003      12.363      40.792
    afdcprc        0.0956     0.0922     1.0370     0.3024     -0.0875      0.2786
    lphypc         0.3081     0.7155     0.4306     0.6678     -1.1127      1.7288
    lbedspc        0.2790     0.5130     0.5438     0.5879     -0.7398      1.2978
    lpcinc        -2.4947     0.9783    -2.5500     0.0124     -4.4374     -0.5520
    lpopul         0.7393     0.7023     1.0526     0.2952     -0.6554      2.1340
    d90            0.5797     0.2761     2.0995     0.0385      0.0314      1.1280
    ==============================================================================


  # part c


```python

exog_vars = ['afdcprc', 'lphypc', 'lbedspc', 'lpcinc', 'lpopul', 'd90']
X = data[exog_vars]
y = data['lowbrth']

model_fd = FirstDifferenceOLS(y, X).fit(cov_type='robust')

print("First Differences Results:")
print(model_fd.summary)
```

    First Differences Results:
                         FirstDifferenceOLS Estimation Summary                      
    ================================================================================
    Dep. Variable:                lowbrth   R-squared:                        0.3839
    Estimator:         FirstDifferenceOLS   R-squared (Between):              0.4490
    No. Observations:                  50   R-squared (Within):               0.3839
    Date:                Sat, May 03 2025   R-squared (Overall):              0.4490
    Time:                        06:09:08   Log-likelihood                   -0.6139
    Cov. Estimator:                Robust                                           
                                            F-statistic:                      4.5700
    Entities:                          50   P-value                           0.0011
    Avg Obs:                       2.0000   Distribution:                    F(6,44)
    Min Obs:                       2.0000                                           
    Max Obs:                       2.0000   F-statistic (robust):             5.3680
                                            P-value                           0.0003
    Time periods:                       2   Distribution:                    F(6,44)
    Avg Obs:                       50.000                                           
    Min Obs:                       50.000                                           
    Max Obs:                       50.000                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    afdcprc       -0.1761     0.0768    -2.2939     0.0266     -0.3308     -0.0214
    lphypc         5.8945     3.0986     1.9023     0.0637     -0.3504      12.139
    lbedspc       -1.5762     1.2362    -1.2750     0.2090     -4.0676      0.9152
    lpcinc        -0.8455     1.4840    -0.5697     0.5717     -3.8364      2.1453
    lpopul         3.4411     2.6877     1.2803     0.2071     -1.9756      8.8578
    d90            0.1060     0.3676     0.2884     0.7744     -0.6348      0.8468
    ==============================================================================


# Part d


```python
data = pd.read_stata('lowbirth.dta')

data['afdcprc_sq'] = data['afdcprc'] ** 2

data = data.set_index(['stateabb', 'year'])


exog_vars = ['afdcprc', 'afdcprc_sq', 'lphypc', 'lbedspc', 'lpcinc', 'lpopul', 'd90']
X = data[exog_vars]
y = data['lowbrth']

model_fd_quad = FirstDifferenceOLS(y, X).fit(cov_type='robust')

print("\nFirst Differences Results with Quadratic Term:")
print(model_fd_quad.summary)

beta_1 = model_fd_quad.params['afdcprc']
beta_6 = model_fd_quad.params['afdcprc_sq']
turning_point = -beta_1 / (2 * beta_6)
print(f"\nTurning Point of Quadratic: {turning_point}")

print(f"Range afdcprc: Min = {data['afdcprc'].min()}, Max = {data['afdcprc'].max()}")
```

    
    First Differences Results with Quadratic Term:
                         FirstDifferenceOLS Estimation Summary                      
    ================================================================================
    Dep. Variable:                lowbrth   R-squared:                        0.4052
    Estimator:         FirstDifferenceOLS   R-squared (Between):             -0.2183
    No. Observations:                  50   R-squared (Within):               0.4052
    Date:                Sat, May 03 2025   R-squared (Overall):             -0.2180
    Time:                        06:40:51   Log-likelihood                    0.2625
    Cov. Estimator:                Robust                                           
                                            F-statistic:                      4.1839
    Entities:                          50   P-value                           0.0014
    Avg Obs:                       2.0000   Distribution:                    F(7,43)
    Min Obs:                       2.0000                                           
    Max Obs:                       2.0000   F-statistic (robust):             5.1245
                                            P-value                           0.0003
    Time periods:                       2   Distribution:                    F(7,43)
    Avg Obs:                       50.000                                           
    Min Obs:                       50.000                                           
    Max Obs:                       50.000                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    afdcprc       -0.5035     0.2612    -1.9276     0.0605     -1.0303      0.0233
    afdcprc_sq     0.0396     0.0318     1.2474     0.2190     -0.0244      0.1036
    lphypc         6.6209     3.4480     1.9202     0.0615     -0.3327      13.574
    lbedspc       -1.4080     1.3441    -1.0475     0.3007     -4.1186      1.3027
    lpcinc        -0.9988     1.5416    -0.6479     0.5205     -4.1077      2.1102
    lpopul         4.4290     2.9252     1.5141     0.1373     -1.4701      10.328
    d90            0.1246     0.3867     0.3222     0.7489     -0.6552      0.9044
    ==============================================================================
    
    Turning Point of Quadratic: 6.355868459603918
    Range afdcprc: Min = 1.0416666269302368, Max = 7.358795166015625


# Question 2

# Part A


```python
data = pd.read_stata('wagepan.dta')

print("Columns:")
print(data.columns)


data = data.set_index(['nr', 'year'])
exog_vars = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87']
X = data[exog_vars]
X = sm.add_constant(X)
y = data['lwage']

pooled_model = sm.OLS(y, X).fit()
print("\nPooled OLS Results:")
print(pooled_model.summary())

re_model = RandomEffects(y, X).fit()
print("\nRandom Effects Results:")
print(re_model.summary)

fe_model = PanelOLS(y, X, entity_effects=True).fit()
print("\nFixed Effects Results:")
print(fe_model.summary)

fd_model = FirstDifferenceOLS(y, data[exog_vars]).fit(cov_type='robust')
print("\nFirst Differences Results:")
print(fd_model.summary)
```

    Columns:
    Index(['nr', 'year', 'agric', 'black', 'bus', 'construc', 'ent', 'exper',
           'fin', 'hisp', 'poorhlth', 'hours', 'manuf', 'married', 'min',
           'nrthcen', 'nrtheast', 'occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6',
           'occ7', 'occ8', 'occ9', 'per', 'pro', 'pub', 'rur', 'south', 'educ',
           'tra', 'trad', 'union', 'lwage', 'd81', 'd82', 'd83', 'd84', 'd85',
           'd86', 'd87', 'expersq'],
          dtype='object')
    
    Pooled OLS Results:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  lwage   R-squared:                       0.075
    Model:                            OLS   Adj. R-squared:                  0.074
    Method:                 Least Squares   F-statistic:                     50.54
    Date:                Sat, 03 May 2025   Prob (F-statistic):           1.43e-69
    Time:                        09:48:37   Log-Likelihood:                -3269.0
    No. Observations:                4360   AIC:                             6554.
    Df Residuals:                    4352   BIC:                             6605.
    Df Model:                           7                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          1.3935      0.022     63.462      0.000       1.350       1.437
    d81            0.1194      0.031      3.845      0.000       0.059       0.180
    d82            0.1782      0.031      5.738      0.000       0.117       0.239
    d83            0.2258      0.031      7.271      0.000       0.165       0.287
    d84            0.2968      0.031      9.558      0.000       0.236       0.358
    d85            0.3459      0.031     11.140      0.000       0.285       0.407
    d86            0.4062      0.031     13.082      0.000       0.345       0.467
    d87            0.4730      0.031     15.232      0.000       0.412       0.534
    ==============================================================================
    Omnibus:                     1026.671   Durbin-Watson:                   0.900
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             6110.791
    Skew:                          -0.988   Prob(JB):                         0.00
    Kurtosis:                       8.453   Cond. No.                         8.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    Random Effects Results:
                            RandomEffects Estimation Summary                        
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.1453
    Estimator:              RandomEffects   R-squared (Between):              0.0000
    No. Observations:                4360   R-squared (Within):               0.1625
    Date:                Sat, May 03 2025   R-squared (Overall):              0.0752
    Time:                        09:48:37   Log-likelihood                   -1660.0
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      105.73
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                  F(7,4352)
    Min Obs:                       8.0000                                           
    Max Obs:                       8.0000   F-statistic (robust):             105.73
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                  F(7,4352)
    Avg Obs:                       545.00                                           
    Min Obs:                       545.00                                           
    Max Obs:                       545.00                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          1.3935     0.0220     63.274     0.0000      1.3503      1.4367
    d81            0.1194     0.0215     5.5609     0.0000      0.0773      0.1615
    d82            0.1782     0.0215     8.2996     0.0000      0.1361      0.2203
    d83            0.2258     0.0215     10.517     0.0000      0.1837      0.2679
    d84            0.2968     0.0215     13.825     0.0000      0.2547      0.3389
    d85            0.3459     0.0215     16.113     0.0000      0.3038      0.3880
    d86            0.4062     0.0215     18.922     0.0000      0.3642      0.4483
    d87            0.4730     0.0215     22.031     0.0000      0.4309      0.5151
    ==============================================================================
    
    Fixed Effects Results:
                              PanelOLS Estimation Summary                           
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.1625
    Estimator:                   PanelOLS   R-squared (Between):           -2.22e-16
    No. Observations:                4360   R-squared (Within):               0.1625
    Date:                Sat, May 03 2025   R-squared (Overall):              0.0752
    Time:                        09:48:37   Log-likelihood                   -1372.4
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      105.56
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                  F(7,3808)
    Min Obs:                       8.0000                                           
    Max Obs:                       8.0000   F-statistic (robust):             105.56
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                  F(7,3808)
    Avg Obs:                       545.00                                           
    Min Obs:                       545.00                                           
    Max Obs:                       545.00                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          1.3935     0.0152     91.715     0.0000      1.3637      1.4233
    d81            0.1194     0.0215     5.5564     0.0000      0.0773      0.1615
    d82            0.1782     0.0215     8.2929     0.0000      0.1361      0.2203
    d83            0.2258     0.0215     10.508     0.0000      0.1837      0.2679
    d84            0.2968     0.0215     13.814     0.0000      0.2547      0.3389
    d85            0.3459     0.0215     16.100     0.0000      0.3038      0.3881
    d86            0.4062     0.0215     18.906     0.0000      0.3641      0.4484
    d87            0.4730     0.0215     22.013     0.0000      0.4309      0.5151
    ==============================================================================
    
    F-test for Poolability: 9.7088
    P-value: 0.0000
    Distribution: F(544,3808)
    
    Included effects: Entity
    
    First Differences Results:
                         FirstDifferenceOLS Estimation Summary                      
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.0252
    Estimator:         FirstDifferenceOLS   R-squared (Between):              0.2709
    No. Observations:                3815   R-squared (Within):               0.1625
    Date:                Sat, May 03 2025   R-squared (Overall):              0.2661
    Time:                        09:48:38   Log-likelihood                   -2308.6
    Cov. Estimator:                Robust                                           
                                            F-statistic:                      14.059
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                  F(7,3808)
    Min Obs:                       8.0000                                           
    Max Obs:                       8.0000   F-statistic (robust):             13.044
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                  F(7,3808)
    Avg Obs:                       545.00                                           
    Min Obs:                       545.00                                           
    Max Obs:                       545.00                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    d81            0.1194     0.0244     4.8969     0.0000      0.0716      0.1672
    d82            0.1782     0.0312     5.7119     0.0000      0.1170      0.2394
    d83            0.2258     0.0353     6.3970     0.0000      0.1566      0.2950
    d84            0.2968     0.0394     7.5384     0.0000      0.2196      0.3740
    d85            0.3459     0.0434     7.9619     0.0000      0.2607      0.4311
    d86            0.4062     0.0475     8.5610     0.0000      0.3132      0.4993
    d87            0.4730     0.0503     9.4091     0.0000      0.3744      0.5716
    ==============================================================================


#Part B


```python
data = pd.read_stata('wagepan.dta')
data = data.set_index(['nr', 'year'])


exog_vars_full = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']
X_full = data[exog_vars_full]
X_pooled = sm.add_constant(X_full)

exog_vars_fe = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87']
X_fe = data[exog_vars_fe]

y = data['lwage']

# 1. Pooled OLS
pooled_model = sm.OLS(y, X_pooled).fit()
print("\nPooled OLS Results:")
print(pooled_model.summary())

# 2. Random Effects (RE)
re_model = RandomEffects(y, X_pooled).fit()
print("\nRandom Effects Results:")
print(re_model.summary)

# 3. Fixed Effects (FE)
fe_model = PanelOLS(y, X_fe, entity_effects=True, drop_absorbed=True).fit()
print("\nFixed Effects Results:")
print(fe_model.summary)
```

    
    Pooled OLS Results:
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  lwage   R-squared:                       0.145
    Model:                            OLS   Adj. R-squared:                  0.143
    Method:                 Least Squares   F-statistic:                     73.66
    Date:                Sat, 03 May 2025   Prob (F-statistic):          7.29e-140
    Time:                        10:16:04   Log-Likelihood:                -3098.3
    No. Observations:                4360   AIC:                             6219.
    Df Residuals:                    4349   BIC:                             6289.
    Df Model:                          10                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.4966      0.057      8.764      0.000       0.386       0.608
    d81            0.1194      0.030      3.997      0.000       0.061       0.178
    d82            0.1782      0.030      5.965      0.000       0.120       0.237
    d83            0.2258      0.030      7.559      0.000       0.167       0.284
    d84            0.2968      0.030      9.937      0.000       0.238       0.355
    d85            0.3459      0.030     11.581      0.000       0.287       0.404
    d86            0.4062      0.030     13.600      0.000       0.348       0.465
    d87            0.4730      0.030     15.835      0.000       0.414       0.532
    educ           0.0771      0.004     17.615      0.000       0.069       0.086
    black         -0.1226      0.024     -5.171      0.000      -0.169      -0.076
    hisp           0.0246      0.021      1.156      0.248      -0.017       0.066
    ==============================================================================
    Omnibus:                     1264.451   Durbin-Watson:                   0.952
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             9464.656
    Skew:                          -1.177   Prob(JB):                         0.00
    Kurtosis:                       9.823   Cond. No.                         108.
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    
    Random Effects Results:
                            RandomEffects Estimation Summary                        
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.1586
    Estimator:              RandomEffects   R-squared (Between):              0.1296
    No. Observations:                4360   R-squared (Within):               0.1625
    Date:                Sat, May 03 2025   R-squared (Overall):              0.1448
    Time:                        10:16:04   Log-likelihood                   -1658.7
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      81.977
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                 F(10,4349)
    Min Obs:                       8.0000                                           
    Max Obs:                       8.0000   F-statistic (robust):             81.977
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                 F(10,4349)
    Avg Obs:                       545.00                                           
    Min Obs:                       545.00                                           
    Max Obs:                       545.00                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    const          0.4966     0.1129     4.4003     0.0000      0.2754      0.7179
    d81            0.1194     0.0215     5.5606     0.0000      0.0773      0.1615
    d82            0.1782     0.0215     8.2992     0.0000      0.1361      0.2203
    d83            0.2258     0.0215     10.516     0.0000      0.1837      0.2679
    d84            0.2968     0.0215     13.824     0.0000      0.2547      0.3389
    d85            0.3459     0.0215     16.112     0.0000      0.3038      0.3880
    d86            0.4062     0.0215     18.921     0.0000      0.3641      0.4483
    d87            0.4730     0.0215     22.030     0.0000      0.4309      0.5151
    educ           0.0771     0.0092     8.3559     0.0000      0.0590      0.0952
    black         -0.1226     0.0500    -2.4529     0.0142     -0.2205     -0.0246
    hisp           0.0246     0.0449     0.5482     0.5836     -0.0634      0.1127
    ==============================================================================
    
    Fixed Effects Results:
                              PanelOLS Estimation Summary                           
    ================================================================================
    Dep. Variable:                  lwage   R-squared:                        0.1625
    Estimator:                   PanelOLS   R-squared (Between):              0.2709
    No. Observations:                4360   R-squared (Within):               0.1625
    Date:                Sat, May 03 2025   R-squared (Overall):              0.2661
    Time:                        10:16:04   Log-likelihood                   -1372.4
    Cov. Estimator:            Unadjusted                                           
                                            F-statistic:                      105.56
    Entities:                         545   P-value                           0.0000
    Avg Obs:                       8.0000   Distribution:                  F(7,3808)
    Min Obs:                       8.0000                                           
    Max Obs:                       8.0000   F-statistic (robust):             105.56
                                            P-value                           0.0000
    Time periods:                       8   Distribution:                  F(7,3808)
    Avg Obs:                       545.00                                           
    Min Obs:                       545.00                                           
    Max Obs:                       545.00                                           
                                                                                    
                                 Parameter Estimates                              
    ==============================================================================
                Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    ------------------------------------------------------------------------------
    d81            0.1194     0.0215     5.5564     0.0000      0.0773      0.1615
    d82            0.1782     0.0215     8.2929     0.0000      0.1361      0.2203
    d83            0.2258     0.0215     10.508     0.0000      0.1837      0.2679
    d84            0.2968     0.0215     13.814     0.0000      0.2547      0.3389
    d85            0.3459     0.0215     16.100     0.0000      0.3038      0.3881
    d86            0.4062     0.0215     18.906     0.0000      0.3641      0.4484
    d87            0.4730     0.0215     22.013     0.0000      0.4309      0.5151
    ==============================================================================
    
    F-test for Poolability: 9.7088
    P-value: 0.0000
    Distribution: F(544,3808)
    
    Included effects: Entity


# Part c


```python

data = pd.read_stata('wagepan.dta')

data = data.set_index(['nr', 'year'])

exog_vars = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']
X = data[exog_vars]
X_pooled = sm.add_constant(X)
y = data['lwage']

# 1. Pooled OLS
pooled_model = sm.OLS(y, X_pooled).fit()

# 2. Random Effects (RE)
re_model = RandomEffects(y, X_pooled).fit()

pooled_se = pooled_model.bse
re_se = re_model.std_errors

se_comparison = pd.DataFrame({
    'Pooled OLS SE': pooled_se,
    'Random Effects SE': re_se
})

print("\ Comparsion Standard Errors:")
print(se_comparison)

se_comparison.to_csv('/content/standard_errors_comparison.txt', sep='\t')


```

    \ Comparsion Standard Errors:
           Pooled OLS SE  Random Effects SE
    const       0.056669           0.112864
    d81         0.029871           0.021471
    d82         0.029871           0.021471
    d83         0.029871           0.021471
    d84         0.029871           0.021471
    d85         0.029871           0.021471
    d86         0.029871           0.021471
    d87         0.029871           0.021471
    educ        0.004377           0.009226
    black       0.023702           0.049967
    hisp        0.021306           0.044914


# Part d


```python
data = pd.read_stata('wagepan.dta')

data = data.set_index(['nr', 'year'])

exog_vars = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']
X = data[exog_vars]
X_pooled = sm.add_constant(X)
y = data['lwage']

pooled_model = sm.OLS(y, X_pooled).fit()

pooled_robust_model = sm.OLS(y, X_pooled).fit(cov_type='HC1')

re_model = RandomEffects(y, X_pooled).fit()

pooled_se = pooled_model.bse
pooled_robust_se = pooled_robust_model.bse
re_se = re_model.std_errors

se_comparison = pd.DataFrame({
    'Pooled OLS SE (Ordinary)': pooled_se,
    'Pooled OLS SE (Robust)': pooled_robust_se,
    'Random Effects SE': re_se
})

print("  comparsion of Standard Errors:")
print(se_comparison)
```

      comparsion of Standard Errors:
           Pooled OLS SE (Ordinary)  Pooled OLS SE (Robust)  Random Effects SE
    const                  0.056669                0.054139           0.112864
    d81                    0.029871                0.032341           0.021471
    d82                    0.029871                0.031185           0.021471
    d83                    0.029871                0.030723           0.021471
    d84                    0.029871                0.032010           0.021471
    d85                    0.029871                0.031758           0.021471
    d86                    0.029871                0.031575           0.021471
    d87                    0.029871                0.030058           0.021471
    educ                   0.004377                0.004182           0.009226
    black                  0.023702                0.024978           0.049967
    hisp                   0.021306                0.020069           0.044914


# Part e


```python
data = pd.read_stata('wagepan.dta')

data = data.set_index(['nr', 'year'])


exog_vars = ['d81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'educ', 'black', 'hisp']
X = data[exog_vars]
X_pooled = sm.add_constant(X)
y = data['lwage']

pooled_robust_model = sm.OLS(y, X_pooled).fit(cov_type='HC1')

re_model = RandomEffects(y, X_pooled).fit()

re_robust_model = RandomEffects(y, X_pooled).fit(cov_type='robust')

pooled_robust_se = pooled_robust_model.bse
re_se = re_model.std_errors
re_robust_se = re_robust_model.std_errors

se_comparison = pd.DataFrame({
    'Pooled OLS SE (Robust)': pooled_robust_se,
    'Random Effects SE (Ordinary)': re_se,
    'Random Effects SE (Robust)': re_robust_se
})

print("\  Comparsion of Standard Errors:")
print(se_comparison)
```

    \  Comparsion of Standard Errors:
           Pooled OLS SE (Robust)  Random Effects SE (Ordinary)  \
    const                0.054139                      0.112864   
    d81                  0.032341                      0.021471   
    d82                  0.031185                      0.021471   
    d83                  0.030723                      0.021471   
    d84                  0.032010                      0.021471   
    d85                  0.031758                      0.021471   
    d86                  0.031575                      0.021471   
    d87                  0.030058                      0.021471   
    educ                 0.004182                      0.009226   
    black                0.024978                      0.049967   
    hisp                 0.020069                      0.044914   
    
           Random Effects SE (Robust)  
    const                    0.103358  
    d81                      0.025471  
    d82                      0.023928  
    d83                      0.023406  
    d84                      0.024600  
    d85                      0.024366  
    d86                      0.024780  
    d87                      0.023479  
    educ                     0.008560  
    black                    0.051718  
    hisp                     0.043128  


# Question 1

# Import Data


```python

data = pd.read_csv('MOM.dat', delim_whitespace=True, names=['lnhr', 'lnwg', 'kids', 'ageh', 'agesq', 'disab', 'id', 'year'])

np.random.seed(123)
random_sample = data.sample(frac=0.5)

print("Data information :")
print(random_sample.info())
print("\n Summary stats:")
print(random_sample[['lnhr', 'lnwg', 'kids', 'ageh', 'agesq', 'disab']].describe())

print("\n Number unique:", random_sample['id'].nunique())
print(" Years:", random_sample['year'].unique())

correlation = random_sample[['lnhr', 'lnwg']].corr()
print("\Correalation between lnhr و lnwg:")
print(correlation)
```

    Data information :
    <class 'pandas.core.frame.DataFrame'>
    Index: 2660 entries, 4914 to 1699
    Data columns (total 8 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   lnhr    2660 non-null   float64
     1   lnwg    2660 non-null   float64
     2   kids    2660 non-null   float64
     3   ageh    2660 non-null   float64
     4   agesq   2660 non-null   float64
     5   disab   2660 non-null   float64
     6   id      2660 non-null   float64
     7   year    2660 non-null   float64
    dtypes: float64(8)
    memory usage: 187.0 KB
    None
    
     Summary stats:
                  lnhr         lnwg         kids         ageh        agesq  \
    count  2660.000000  2660.000000  2660.000000  2660.000000  2660.000000   
    mean      7.662297     2.617041     1.552632    39.036090  1596.993233   
    std       0.286114     0.424661     1.186871     8.555958   701.079178   
    min       2.770000    -0.260000     0.000000    22.000000   484.000000   
    25%       7.580000     2.370000     1.000000    32.000000  1024.000000   
    50%       7.650000     2.640000     2.000000    38.000000  1444.000000   
    75%       7.780000     2.860000     2.000000    45.000000  2025.000000   
    max       8.540000     4.650000     6.000000    60.000000  3600.000000   
    
                 disab  
    count  2660.000000  
    mean      0.060902  
    std       0.239196  
    min       0.000000  
    25%       0.000000  
    50%       0.000000  
    75%       0.000000  
    max       1.000000  
    
     Number unique: 532
     Years: [1983. 1979. 1987. 1984. 1980. 1985. 1986. 1982. 1988. 1981.]
    \Correalation between lnhr و lnwg:
              lnhr      lnwg
    lnhr  1.000000  0.137905
    lnwg  0.137905  1.000000


    <ipython-input-5-5f2422162a61>:1: FutureWarning: The 'delim_whitespace' keyword in pd.read_csv is deprecated and will be removed in a future version. Use ``sep='\s+'`` instead
      data = pd.read_csv('MOM.dat', delim_whitespace=True, names=['lnhr', 'lnwg', 'kids', 'ageh', 'agesq', 'disab', 'id', 'year'])


# Part b


```python


def panel_bootstrap(formula, data, n_boot=200, cluster_var='id'):
    try:
        coefs = []
        ids = data[cluster_var].unique()
        for _ in tqdm(range(n_boot), desc="Bootstrap"):
            sampled_ids = np.random.choice(ids, size=len(ids), replace=True)
            boot_data = []
            for sid in sampled_ids:
                boot_data.append(data[data[cluster_var] == sid])
            boot_data = pd.concat(boot_data)
            model = smf.ols(formula, data=boot_data).fit()
            coefs.append(model.params.get('lnwg', model.params.get('lnwg_diff', np.nan)))
        se_boot = np.std(coefs, ddof=1)
        return se_boot if not np.isnan(se_boot) else np.nan
    except Exception as e:
        print(f"Error in bootstrap: {e}")
        return np.nan

results = []

# 1. Pooled OLS
print("\n1. Pooled OLS")
try:
    pooled_ols = smf.ols('lnhr ~ lnwg', data=random_sample).fit(cov_type='cluster', cov_kwds={'groups': random_sample['id']})
    beta_ols = pooled_ols.params['lnwg']
    se_ols = pooled_ols.bse['lnwg']
    se_boot_ols = panel_bootstrap('lnhr ~ lnwg', random_sample)
    results.append(['Pooled OLS', beta_ols, se_ols, se_boot_ols])
except Exception as e:
    print(f" Error in Pooled OLS: {e}")
    results.append(['Pooled OLS', np.nan, np.nan, np.nan])

# 2. Between Estimator
print("\n2. Between Estimator")
try:
    between_data = random_sample.groupby('id')[['lnhr', 'lnwg']].mean().reset_index()
    between_model = smf.ols('lnhr ~ lnwg', data=between_data).fit()
    beta_between = between_model.params['lnwg']
    se_between = between_model.bse['lnwg']
    se_boot_between = panel_bootstrap('lnhr ~ lnwg', between_data)
    results.append(['Between', beta_between, se_between, se_boot_between])
except Exception as e:
    print(f"Error in  Between: {e}")
    results.append(['Between', np.nan, np.nan, np.nan])

# 3. Within (Fixed Effects)
print("\n3. Within (Fixed Effects)")
try:
    within_model = smf.ols('lnhr ~ lnwg + C(id)', data=random_sample).fit(cov_type='cluster', cov_kwds={'groups': random_sample['id']})
    beta_within = within_model.params['lnwg']
    se_within = within_model.bse['lnwg']
    se_boot_within = panel_bootstrap('lnhr ~ lnwg + C(id)', random_sample)
    results.append(['Within', beta_within, se_within, se_boot_within])
except Exception as e:
    print(f" Error in  Within: {e}")
    results.append(['Within', np.nan, np.nan, np.nan])

# 4. First Differences
print("\n4. First Differences")
try:
    fd_data = random_sample.sort_values(['id', 'year']).copy()
    fd_data['lnhr_diff'] = fd_data.groupby('id')['lnhr'].diff()
    fd_data['lnwg_diff'] = fd_data.groupby('id')['lnwg'].diff()
    fd_data = fd_data.dropna()
    fd_model = smf.ols('lnhr_diff ~ lnwg_diff - 1', data=fd_data).fit(cov_type='cluster', cov_kwds={'groups': fd_data['id']})
    beta_fd = fd_model.params['lnwg_diff']
    se_fd = fd_model.bse['lnwg_diff']
    se_boot_fd = panel_bootstrap('lnhr_diff ~ lnwg_diff - 1', fd_data)
    results.append(['First Differences', beta_fd, se_fd, se_boot_fd])
except Exception as e:
    print(f" Error in  First Differences: {e}")
    results.append(['First Differences', np.nan, np.nan, np.nan])

# 5. Random Effects GLS
print("\n5. Random Effects GLS")
try:
    re_gls = smf.mixedlm('lnhr ~ lnwg', data=random_sample, groups=random_sample['id']).fit()
    beta_re_gls = re_gls.params['lnwg']
    se_re_gls = re_gls.bse['lnwg']
    ols_for_re = smf.ols('lnhr ~ lnwg', data=random_sample).fit(cov_type='cluster', cov_kwds={'groups': random_sample['id']})
    se_boot_re_gls = ols_for_re.bse['lnwg']
    results.append(['Random Effects GLS', beta_re_gls, se_re_gls, se_boot_re_gls])
except Exception as e:
    print(f" Error in  Random Effects GLS: {e}")
    results.append(['Random Effects GLS', np.nan, np.nan, np.nan])

# 6. Random Effects MLE
print("\n6. Random Effects MLE")
try:
    re_mle = smf.mixedlm('lnhr ~ lnwg', data=random_sample, groups=random_sample['id']).fit()
    beta_re_mle = re_mle.params['lnwg']
    se_re_mle = re_mle.bse['lnwg']
    se_boot_re_mle = ols_for_re.bse['lnwg']
    results.append(['Random Effects MLE', beta_re_mle, se_re_mle, se_boot_re_mle])
except Exception as e:
    print(f" Error in  Random Effects MLE: {e}")
    results.append(['Random Effects MLE', np.nan, np.nan, np.nan])

print("\n Estimators results:")
results_df = pd.DataFrame(results, columns=['Estimator', 'Beta', 'Default SE', 'Bootstrap SE'])
print(results_df.round(4))

results_df.to_csv('panel_estimates.csv', index=False)
```

    
    1. Pooled OLS


    Bootstrap: 100%|██████████| 200/200 [00:37<00:00,  5.37it/s]


    
    2. Between Estimator


    Bootstrap: 100%|██████████| 200/200 [00:34<00:00,  5.79it/s]


    
    3. Within (Fixed Effects)


    Bootstrap: 100%|██████████| 200/200 [01:25<00:00,  2.35it/s]


    
    4. First Differences


    Bootstrap: 100%|██████████| 200/200 [00:35<00:00,  5.59it/s]


    
    5. Random Effects GLS
    
    6. Random Effects MLE
    
    نتایج تخمین‌ها:
                Estimator    Beta  Default SE  Bootstrap SE
    0          Pooled OLS  0.0929      0.0358        0.0359
    1             Between  0.0811      0.0207        0.0292
    2              Within  0.1526      0.1531        0.1247
    3   First Differences  0.2249      0.1762        0.1683
    4  Random Effects GLS  0.1073      0.0167        0.0358
    5  Random Effects MLE  0.1073      0.0167        0.0358


```python

```
