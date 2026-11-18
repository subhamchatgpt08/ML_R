# import numpy as np
# import pandas as pd 
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# df = pd.read_csv("data/Churn_Modelling.csv")
# print(df.head())

# # remove the columns 
# df.drop(["RowNumber", "CustomerId", "Surname"], axis=1 , inplace=True)

# # le = LabelEncoder()
# # df["Gender"] = le.fit_transform(df["Gender"])

# # scaler = MinMaxScaler()

# # df_scaled = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

# # print(df_scaled)


import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. Load Dataset
df = pd.read_csv("data/Churn_Modelling.csv")

# 2. Drop Unnecessary Columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# 3. Label Encoding
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

# 4. One Hot Encoding
df = pd.get_dummies(df, columns=["Geography"], dtype=int)

# 5. Select Numerical Columns
columns = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary"
]

# 6. Apply MinMaxScaler
scaler = MinMaxScaler()

df[columns] = scaler.fit_transform(df[columns])

# 7. Display DataFrame
print(df.head())

# 8. Save CSV
df.to_csv("Churn_Encoded_Scaled.csv", index=False)

print("Encoding and Scaling Completed Successfully!")