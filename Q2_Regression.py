import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor



#importing the data
data = pd.read_csv('insurance.csv')
charges_range = pd.Series(data['charges'])
print(f'charges range from {round(charges_range.min(),2)} to {round(charges_range.max(),2)}')

#preprocessing tha data
print(data.shape)
print(data.duplicated())
print("\nMissing Values\n",data.isnull().sum())
print('Type of Columns\n',data.dtypes)


#one hot-encoding for the cols that contain categorical values
data = pd.get_dummies(data,columns = ['sex','region','smoker'],prefix = ['sex','region','smoker'])


#seperating the column of the target from the rest data
x = data.drop('charges',axis = 1)
y = data['charges']


#Splitting the Data
perc = 0.2
x_train, x_test, y_train ,y_test = train_test_split(x , y, test_size = perc , random_state = 42)


#Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#Regression Evaluation Function
def printCoefficients (y_test, y_pred, regrCoeff, regrIntercept):
    print ('\n')
    print('Coefficients: \n', regrCoeff)
    print ('Intercept: \n',regrIntercept)
    # The mean squared error
    print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
    print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f' % r2_score(y_test, y_pred))


#Linear Regression
regr = (linear_model.LinearRegression())
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)


#Evaluation linear Regression Model
printCoefficients(y_test, y_pred, regr.coef_, regr.intercept_)

#Gridsearch for the best a value
param_grid = {'alpha' : [122,123,124,125,126]}

#regression Lasso Gridsearch
regrLasso = linear_model.Lasso()
grid_searchLasso = GridSearchCV(estimator=regrLasso, param_grid=param_grid, scoring='r2',cv=5,n_jobs=-1)
grid_searchLasso.fit(x_train, y_train)

#best lasso model
bestLasso = grid_searchLasso.best_estimator_
y_pred_lasso = grid_searchLasso.predict(x_test)

print("Best alpha value (Lasso):", grid_searchLasso.best_params_)

printCoefficients(y_test, y_pred_lasso, bestLasso.coef_, bestLasso.intercept_)
#we accomplish the same R2 value while having less information(Due to the Feature Selection of Lasso Regression many stats drop to 0)
#So less expensive computational but same R2 results.

#coefficients that become 0 from lasso feature selection
feature_names = x.columns
df_coefs = pd.DataFrame({'Feature':feature_names,'Coefficient':bestLasso.coef_})

zeroed = df_coefs[df_coefs['Coefficient']==0]
print("Features that Lasso eliminated:")
print(zeroed)



#Poly
degrees = [1,2,3]
results = {}
for d in degrees:
    #we create a linear regression object
    regrPoly = linear_model.LinearRegression()

    poly = PolynomialFeatures(degree=d)

    # transform the data to make the suitable for polynomial regression
    polyFeaturesTrain = poly.fit_transform(x_train)
    polyFeaturesTest = poly.transform(x_test)

    #using lasso at the new Poly Features
    lasso_poly = linear_model.Lasso(alpha=123,max_iter=100000)
    lasso_poly.fit(polyFeaturesTrain, y_train)

    y_pred_LassoPoly = lasso_poly.predict(polyFeaturesTest)
    print(f"for degree: {d} we had R2 score {round(r2_score(y_test, y_pred_LassoPoly),2)}")


#Best Model using the degree with the most R2 value
regrPoly_best = linear_model.LinearRegression()

poly = PolynomialFeatures(degree=2)
polyFeaturesTrain = poly.fit_transform(x_train)
polyFeaturesTest = poly.transform(x_test)
lasso_poly_best = linear_model.Lasso(alpha=123,max_iter=100000)#maximum tries of the model(mat_iter)
lasso_poly_best.fit(polyFeaturesTrain, y_train)
y_pred_lasso_poly = lasso_poly_best.predict(polyFeaturesTest)


printCoefficients(y_test,y_pred_lasso_poly,lasso_poly_best.coef_, lasso_poly_best.intercept_)

# Getting the names of the new polynomials features
poly_features_names = poly.get_feature_names_out(x.columns)

#Building df polynomial features degree 2
df_poly_coefs = pd.DataFrame({
    'Feature': poly_features_names,
    'Coefficient': lasso_poly_best.coef_
})

#Finding which coefficients from the 91 coefficients became 0
zeroed_poly = df_poly_coefs[df_poly_coefs['Coefficient'] == 0]

#we keep only the important features
kept_poly = df_poly_coefs[df_poly_coefs['Coefficient'] != 0].sort_values(by='Coefficient', ascending=False)

print(f"\n--- Polynomial Lasso Analysis (Degree 2) ---")
print(f"Total Features created: {len(poly_features_names)}")
print(f"Features eliminated (set to 0): {len(zeroed_poly)}")
print(f"Features kept: {len(kept_poly)}")
print("\nTop 10 most influential features/interactions:")
print(kept_poly.head(10))


# Calculate VIF

x_vif = x.astype(float)


vif = pd.DataFrame()
vif["Features"] = x.columns
vif["VIF"] = [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])]

print(vif)
#Searching the correlation between data using vif , the results are excellent because age bmi and children are closes to 1
#while everything else is close to inf , but that is smthing we expect because of the get dummies e.g. if sb is not male then it is a female
#that is why the model is certain for dependency


#plot for linear model
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='gray', alpha=0.5, label='Simple Linear Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges (Simple Linear Regression)')
plt.legend()
plt.show()

#plot for Lasso Poly regression model
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_lasso_poly, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Insurance Charges (Lasso Poly Degree 2)')
plt.show()















