import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

#10-nn impuatation
#https://www.researchgate.net/publication/220981745_A_Study_of_K-Nearest_Neighbour_as_an_Imputation_Method

data = {
    'Feature1': [5, 7, 6, np.nan, 8, 10, np.nan, 12, 14, np.nan],
    'Feature2': [100, 120, np.nan, 105, 115, np.nan, 140, 135, 160, 170],
    'Feature3': [np.nan, 30, 25, 28, np.nan, 50, 55, np.nan, 65, 70]
}

df = pd.DataFrame(data)

#linear regresion
def linear_regression_impute(df, target_column):
    temp_df = df.copy()

    known_data = temp_df.dropna(subset=[target_column])  # แถวที่ target_column ไม่ใช่ NaN
    missing_data = temp_df[temp_df[target_column].isna()]  # แถวที่ target_column เป็น NaN

    if missing_data.empty:
        return df

    feature_columns = [col for col in df.columns if col != target_column]

    known_data = known_data.dropna(subset=feature_columns)

    if known_data.empty:
        return df

    X_train = known_data[feature_columns]
    y_train = known_data[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = missing_data[feature_columns]
    X_missing = X_missing.dropna()  # ลบแถวที่ยังมีค่า NaN ออกจากฟีเจอร์ที่ใช้พยากรณ์

    if not X_missing.empty:
        y_pred = model.predict(X_missing)
        temp_df.loc[X_missing.index, target_column] = y_pred

    return temp_df


for feature in df.columns:
    df = linear_regression_impute(df, feature)


#mean
df['Feature1'] = df['Feature1'].fillna(df['Feature1'].mean())

#knn
imputer = KNNImputer(n_neighbors=10)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(data)


# series data
# Linear Interpolation
df['Linear_Interpolated'] = df['Value'].interpolate(method='linear')

# Polynomial Interpolation (Degree 2)
df['Polynomial_Interpolated'] = df['Value'].interpolate(method='polynomial', order=2)

# Spline Interpolation (Degree 2)
df['Spline_Interpolated'] = df['Value'].interpolate(method='spline', order=2)





