import pandas as pd
import statsmodels.api as sm

myData = pd.read_csv('test_random data.csv')

# add intercept
myData['intercept'] = 1.0

# calculate mediators
myData['M1'] = myData.iloc[:, 0:3].mean(axis=1).astype(float)
myData['M2'] = myData.iloc[:, 3:7].mean(axis=1).astype(float)

# convert 'Gendered_AI' and ensure dummies are numeric
gendered_ai = pd.get_dummies(myData['Gendered_AI'], prefix='AI').astype(float)

# add gender_ai to data
myData = pd.concat([myData, gendered_ai], axis=1)

# check DV values are all numeric
myData['DV'] = myData['DV'].astype(float)

# IV into a list
IVs = ['intercept', 'AI_M', 'AI_F', 'AI_N']

# convert any non-numeric data in predictors and DV
for c in IVs + ['M1', 'M2', 'DV']:
    if myData[c].dtype != 'float64':
        print(f"Column {c} is not float64, but {myData[c].dtype}")


# IV on DV
model_0 = sm.OLS(myData['DV'], myData[IVs]).fit()

# IV on M1
model_m1 = sm.OLS(myData['M1'], myData[IVs]).fit()

# IV on M2
model_m2 = sm.OLS(myData['M2'], myData[IVs]).fit()

# M1 on DV
model_m1_y = sm.OLS(myData['DV'], myData[['intercept', 'M1']]).fit()

# M2 on DV
model_m2_y = sm.OLS(myData['DV'], myData[['intercept', 'M2']]).fit()

# IV + M1 + M2 on DV
full_model_predictors = IVs + ['M1', 'M2']
model_y = sm.OLS(myData['DV'], myData[full_model_predictors]).fit()

print("IV on DV Model Summary:")
print(model_0.summary())

print("\nIV on M1 Model Summary:")
print(model_m1.summary())

print("\nIV on M2 Model Summary:")
print(model_m2.summary())

print("\nMediator 1 on DV Model Summary:")
print(model_m1_y.summary())

print("\nMediator 2 on DV Model Summary:")
print(model_m2_y.summary())

print("\nIV + M1 + M2 on DV Model Summary:")
print(model_y.summary())
