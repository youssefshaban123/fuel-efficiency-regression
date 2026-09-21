What's in this project
Regression_Analysis_Final_Project.ipynb – the notebook with all the code and analysis
auto_mpg_dataset.csv – the dataset (398 cars from the 70s-80s, with specs like weight, horsepower, cylinders, etc.)
What I did
Cleaned the data (there were some missing horsepower values and some outliers, so I handled both)
Did some EDA – histograms, boxplots, a correlation heatmap, pairplots
Built a simple linear regression model from scratch (just using the OLS formulas, no sklearn) using weight as the predictor since it had the strongest correlation with mpg
Built a multiple linear regression model using sklearn, with several features plus origin (one-hot encoded). Had to check for multicollinearity first (VIF) and dropped displacement because it was too correlated with the other features
Compared the two models
Checked the regression assumptions (linearity, independence, homoscedasticity, normality of residuals)
Tried to improve the model with a log transform of the target, plus Ridge and Lasso regression
Results

The strongest single predictor of mpg turned out to be weight (correlation of -0.83).

SLR equation: mpg = 45.98 - 0.0076 * weight

Model	Test MSE	Test R²
SLR (weight only)	15.86	0.728
MLR	10.24	0.825
MLR (log target)	8.04	0.862
Ridge	10.31	0.823
Lasso	10.45	0.821

The MLR model did a lot better than the simple one (about 35% lower error), which makes sense since mpg depends on more than just one factor. Log-transforming the target helped even more, probably because the Breusch-Pagan test showed the plain model had some heteroscedasticity issues. Ridge and Lasso didn't really improve things here, so overfitting wasn't really a problem after I removed the collinear feature.

Assumption checks
Independence: Durbin-Watson was 2.175, so no real autocorrelation issue
Homoscedasticity: Breusch-Pagan test came back significant (p = 0.0017), so this assumption was violated a bit — that's part of why I tried the log transform
Normality: Shapiro-Wilk also came back significant (p = 0.0046), so residuals aren't perfectly normal but not too bad
How to run it
bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn

Then just open the notebook and run all the cells.
