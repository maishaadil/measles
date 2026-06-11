# measles

## 1. Intended Use of the Predictor
This predictor is designed as a prospective surveillance tool for national public health agencies and the European Centre for Disease Prevention and Control (ECDC). Given a country’s MMR vaccination coverage, incoming air travel volume, and the prior month’s confirmed case count, the model forecasts monthly measles incidence at the country level.

The primary use case is risk stratification. The model is designed to flag countries at elevated incidence risk before large outbreaks become apparent in surveillance data. This is particularly relevant for countries with declining vaccination coverage, such as Montenegro, where first-dose coverage fell from 91% in 2011 to 18% in 2021.

One important limitation is that the prior month’s case count (counts_lag1) is the model’s strongest predictor. This makes the model well-suited for tracking ongoing outbreaks but less useful for early warning before an initial case cluster appears. Agencies should treat the model as a complement to, not a substitute for, real-time surveillance systems.

## 2. Study Design and Impact on Development
Data come from Romano et al. (2024); the authors combine epidemiological surveillance and air travel records for 32 European countries from 2011 to 2023. The unit of analysis is the country-month. After combining measles and flight data, the panel contains 4,992 rows (32 countries * 13 years * 12 months).

### Outcome
The outcome is log-transformed monthly measles incidence: log(cases + 1). Raw monthly case counts are extremely right-skewed (skewness = 13.74) and zero-inflated, with approximately 59% of country-months reporting zero cases. The log transformation reduces skewness to 1.71 and is used as the outcome in all five candidate models. Figures 1 and 2 show the distributions of raw and log-transformed case counts. 

### Flight Data Imputation
Observed flight data are available for 2019, 2020, and 2022 only. Years outside this range were imputed as follows: 2011-2018 were assigned the 2019 observed value as a constant pre-COVID baseline; 2021 was interpolated as the average of 2020 and 2022; and 2023 carried the 2022 value forward. Fitting a trend through three anchor years, one of which reflects COVID-era suppression, would produce implausible pre-2019 estimates, so the constant baseline was preferred.

### MCV2 Imputation
Second-dose MMR coverage (MCV2) was missing for 21 country-years and imputed in two stages. Finland, Italy, and Luxembourg had partial data and were filled using within-country linear interpolation. Ireland had no observed MCV2 in any year and was handled separately: MCV2 was regressed on MCV1 and Year using the 403 country-years with observed values, and the fitted values were applied to Ireland’s 13 rows. The model produced estimates of 88-90%, within the European interquartile range. All imputed values are flagged with a binary indicator in the dataset; this indicator is not used as a model feature.

### Panel Design Limitation on Vaccination Effects
Country fixed effects absorb the between-country MCV1 variation that drives the visible relationship between coverage and incidence in raw data. After conditioning on country, within-country signal is insufficient to identify a nonlinear herd-immunity threshold. Figures 3 and 4 show the coverage distributions; Figure 5 visualizes longitudinal case data by country.

### Evaluation Strategy
All five candidate models were trained on 2011-2021 data (4,192 rows) and evaluated on a held-out 2022-2023 test set (768 rows). Temporal ordering is fully preserved; models never see future data during training. This split mirrors the intended prospective use of the predictor and provides a clean, unambiguous estimate of out-of-sample performance for each model.

## 3. Methodology
Five candidate models were estimated and compared: linear regression, lasso, a generalized additive model (GAM), random forest, and SuperLearner. All five models use the same eight features (MCV1, MCV2, log_pop, log_flights, counts_lag1, Year, month, country) so that differences in test MSE reflect differences in model flexibility, not differences in input information.

### Linear Regression
OLS linear regression with country and month as fixed effects serves as the baseline, establishing a performance floor for the comparison. Partial effect estimates for each continuous predictor are shown in Figures 6-9.

### Lasso
Lasso (alpha = 1 in glmnet) was included for coefficient shrinkage and feature selection. Lambda was selected via 10-fold cross-validation on the training set. No features were zeroed out, confirming the eight-feature set is lean. MCV2 received a positive coefficient due to collinearity with MCV1 rather than a true association; these two coefficients should not be interpreted in isolation. Figure 10 shows the non-zero coefficients from the full-dataset fit.

### Generalized Additive Model
The GAM was initially specified with smooth terms for MCV1 and MCV2 to test for a nonlinear herd-immunity threshold near 95% first-dose coverage. The smooth GAM performed worse than OLS, and the MCV1 partial effect was almost linear, consistent with the panel design limitation discussed in Section 2. The GAM was re-specified with all linear terms, producing identical test MSE to OLS. Figures 11-14 show the partial effects.

### Random Forest
Random forest was included to capture nonlinear interactions and heterogeneous feature effects without requiring a prespecified functional form. Five hundred trees were fit using the R randomForest package, with mtry tuned on the training set via tuneRF (details in Section 4). Random forest was the best-performing model, reducing test RMSE by 48% relative to the linear models.

### SuperLearner
SuperLearner constructs a weighted ensemble of candidate learners. Ensemble weights are estimated via cross-validated NNLS on the training set. The candidate library included eight learners: an intercept-only mean, linear regression, ridge, elastic net, lasso, a GAM with factor-to-integer conversion for month and country, and two random forest candidates (default mtry and mtry = 2). Training-fit weights concentrated on the two random forest candidates (0.677 and 0.291), with a small residual weight on the GAM (0.033) and all other learners zeroed out. The ensemble test MSE (0.3364) was marginally higher than the standalone random forest (0.3141), because the small weight assigned to the GAM pulled predictions slightly away from pure random forest.

### Note on log_flights
Flight volume is included in all five models to maintain a consistent feature set. The within-country variation in log_flights in the training data comes almost entirely from the COVID-era contrast: the decline in 2020 and recovery in 2021. This variation coincided with broader pandemic disruptions to measles transmission, healthcare utilization, and case reporting through mechanisms unrelated to air travel. In the linear regression and GAM, the log_flights coefficient conflates these effects and should not be interpreted as the causal effect of travel on incidence (Figures 9 and 14). RF variable importance (Figure 15) is preferred for assessing the independent predictive contribution of log_flights.

## 4. Loss Function, Hyperparameter Tuning, and Performance Evaluation
### Loss Function
Mean squared error (MSE) on the log(cases + 1) scale was used as the loss function for all five models. MSE is the standard loss for continuous regression and penalizes large prediction errors more heavily than small ones. Using the log scale compresses the extreme right tail of the case count distribution, making MSE a more stable and consistent metric across country-months with very different case magnitudes.

### Hyperparameter Tuning
Lasso: The regularization parameter lambda was selected via 10-fold cross-validation on the 2011-2021 training set using cv.glmnet, choosing lambda.min. The 2022-2023 test set was not used during this step.
Random forest: The number of features sampled at each node (mtry) was tuned on the training set using tuneRF with OOB error as the criterion. Starting at mtry = 2 (floor of 8 features / 3), stepping to mtry = 3 marginally reduced OOB error and was selected for the training-set fit used to compute test MSE. A separate full-dataset fit, run after evaluation and used solely for the variable importance plot (Figure 15), independently selected mtry = 2.
SuperLearner: Ensemble weights were estimated by minimizing the 10-fold cross-validated MSE on the training set using NNLS, which constrains weights to be non-negative and sum to one. 

### Performance Evaluation
Test MSE was computed on the 2022-2023 holdout for all five models using the same split and feature set, enabling a direct comparison across model classes. Test RMSE is reported alongside MSE because it is on the same scale as the outcome, making the magnitude of prediction error easier to interpret.

Model
Test MSE
Test RMSE
Linear Regression
1.1615
1.0777
Lasso
1.1320
1.0639
GAM (linearized)
1.1615
1.0777
Random Forest
0.3141
0.5605
SuperLearner
0.3364
0.5800

Random forest is the best-performing model, reducing test RMSE by approximately 48% relative to the linear models (1.078 to 0.561 log-units). On the raw case count scale, this corresponds to a typical prediction error of roughly 1.75 times the observed count (e0.561 = 1.75). SuperLearner is competitive but does not improve on the standalone random forest. Figures 16 and 17 show predicted vs. observed and Bland-Altman diagnostics for the winning model.

## 5. Feature Engineering and Exploratory Data Analysis

### Feature Engineering
Three log transformations were applied: log(cases + 1) as the outcome, log(pop) as a population size feature, and log(flights + 1) as the travel volume feature. All three variables are right-skewed in raw form, and the log scale reduces leverage from extreme values.

A one-month lagged case count (counts_lag1) was constructed by sorting the panel by country and time and assigning each row the previous row’s case count within the same country. January 2011 observations have no prior month and were assigned NA; these 32 rows were excluded at the modeling step rather than filled with zero, since December 2010 case counts are unknown.

Country and month were coerced to unordered factors. Linear regression and GAM expand these to dummy variables internally; random forest uses them as factors directly. For the GAM inside SuperLearner, which requires numeric inputs, both were converted to integer level codes.

Year was included as a continuous numeric variable. A binary COVID-19 disruption indicator for 2020-2021 was considered but excluded in favour of numeric Year, which captures the same long-term temporal trend without being defined by only two years of data.

### Exploratory Data Analysis
Figure 1 shows the distribution of raw monthly case counts. The distribution is extremely right-skewed with a large mass at zero, confirming the need for the log transformation. Figure 2 shows the distribution after applying log(cases + 1), with skewness reduced from 13.74 to 1.71.

Figures 3 and 4 show the distributions of MCV1 and MCV2 coverage across 416 country-years (32 countries * 13 years), with a reference line at the 95% herd-immunity threshold. The majority of European countries maintain coverage near or above this threshold. Montenegro is the visible outlier in the MCV1 distribution at the far left of the range.

Figure 5 shows monthly case counts over time for all 32 countries. Outbreak activity is clustered, with a small number of country-year combinations accounting for most observed cases. The 2020-2021 period shows suppressed counts across nearly all countries, consistent with COVID-era reductions in transmission and reporting.

## 6. Graphical Displays
All figures are produced in the accompanying R Markdown notebook. Figure numbers match the order in which plots appear in the knitted output.

### Figure 6-9: Linear Regression Partial Effects
Figures 6-9 show the partial effects of each continuous predictor on log(cases + 1) from the full-dataset linear regression fit. Figure 6 shows a negative relationship between MCV1 coverage and incidence, reflecting the protective effect of first-dose vaccination after conditioning on the country. Figure 7 shows the partial effect of MCV2 on incidence. The direction of the estimate should be interpreted cautiously given the collinearity between MCV1 and MCV2 discussed in Section 3. Figure 8 shows the positive relationship between prior month case count and incidence, capturing epidemic momentum. Figure 9 shows the partial effect of log_flights; as discussed in Section 3, this estimate is confounded by COVID-era disruptions and should not be interpreted as the causal effect of air travel on incidence.

### Figure 10: Lasso Coefficients
Figure 10 shows the non-zero coefficients from the full-dataset lasso fit, excluding country and month indicators. Green bars indicate positive coefficients; burgundy bars indicate negative coefficients. All eight features were retained after regularization, confirming the feature set is lean. The positive MCV2 coefficient is a collinearity artifact rather than a biological finding, as discussed in Section 3.

### Figures 11-14: GAM Partial Effects
Figures 11-14 mirror Figures 6-9 for the full-dataset GAM fit. Because the GAM was re-specified with all linear terms, the partial effect plots are visually similar to their linear regression counterparts. The near-linear MCV1 partial effect in Figure 11 confirms that smooth terms do not improve on the linear specification after conditioning on country.

### Figure 15: Random Forest Feature Importance
Figure 15 shows permutation-based feature importance (%IncMSE, the percent increase in MSE when a feature is randomly shuffled) from the full-dataset random forest fit. Prior month case count dominates by a wide margin, followed by country and Year, with MCV1 and log_flights contributing meaningfully as well. Permutation-based importance is used because it directly measures each feature's contribution to prediction error, making it more interpretable than alternative importance metrics.

## Figures 16-17: Model Evaluation Diagnostics
Random forest was the best-performing model and was selected for diagnostic evaluation on the 2022-2023 holdout. Figure 16 plots predicted against observed log(cases + 1). Points clustering near the 45-degree reference line indicate good overall calibration, although the model under-predicts at the highest observed values. Figure 17 is a Bland-Altman plot, which plots the difference between predicted and observed values against their average, making it easier to detect systematic error across the prediction range. Positive values on the y-axis indicate over-prediction. Bias is near zero (mean difference = 0.007 log-units), with limits of agreement of -1.092 to 1.106, and error variance increases with case magnitude.

## 7. Implementation and Monitoring Plan

### Deployment
Under a hypothetical deployment to ECDC or national agencies, the model would operate on a monthly refresh cycle. Each month, the agency would provide updated inputs: MCV1 and MCV2 from WHO European vaccination monitoring reports, incoming air passenger counts from Eurostat, and the prior month’s confirmed case count from ECDC surveillance. The model would produce a predicted log(cases + 1) for each country in the coming month. For operational use, predictions would be back-transformed and flagged when predicted incidence exceeds a country-specific threshold. The exact threshold would be set by the agency based on their risk tolerance, though a reasonable starting point is the 90th percentile of each country’s historical non-zero monthly counts.

### Feature Availability and Drift
Three features require monitoring for availability and drift at deployment. MCV1 and MCV2 are reported with a one to two year lag in WHO country reports; the most recently available estimate would be carried forward in real time. Eurostat flight data are delayed by three to six months and would be handled the same way. counts_lag1 is the most sensitive feature because it depends on timely case reporting. Surveillance gaps like those observed in 2020-2021 degrade it most severely, and predictions from affected months should be flagged and interpreted with caution.

### Recalibration
The model should be recalibrated when any of the following conditions are met. First, if rolling six-month residual monitoring detects sustained prediction bias, the model should be refit on more recent data. Second, a structural shift in the MCV1 distribution, such as a large-scale vaccine hesitancy event or policy change, would require retraining to capture the new coverage landscape. Third, the addition of a new country to the European surveillance system would require retraining to include a fixed effect for that country.

### Limitations
Several limitations should be noted. The model’s reliance on counts_lag1 as its strongest predictor makes it well-suited for tracking ongoing outbreaks but less useful for early warning before an initial case cluster appears. Country fixed effects absorb between-country vaccination coverage variation, making it impossible to detect a within-country herd-immunity threshold from this panel design. The continuous regression outcome also does not explicitly account for zero-inflation; approximately 59% of country-months report zero cases, and predictions for these country-months are compressed near zero by the log transform and contribute little to overall RMSE. The unit of analysis is the country-month, so predictions cannot be disaggregated below the national level and do not identify local or provincial outbreak locations. The model was trained on European data from 2011 to 2023 and should not be applied to other geographic regions without retraining. Finally, the travel volume signal in the training data is identified primarily through COVID-era variation, which coincided with broader pandemic disruptions unrelated to air travel. As a result, the log_flights coefficient in the linear models is not causally interpretable, and Figure 15 provides a more reliable basis for assessing the contribution of travel volume than coefficient-based explanations.

## 8. References

Branda, F., & Romano, C. (2024). Measles (Version v2) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.13748895

Romano, C., Branda, F., Scarpa, F., Jona Lasinio, G., & Ciccozzi, M. (2024). Monitoring measles infections using flight passenger dynamics in Europe: A data-driven approach. Scientific Data, 11, 1358. https://doi.org/10.1038/s41597-024-04231-x
