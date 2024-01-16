"""각기 다른 버전의 eda 모듈"""

import pandas as pd
import numpy as np
import seaborn as sns

def preprocess():
    """Function printing python version."""

    df = pd.read_csv("https://bit.ly/telco-csv", index_col="customerID")
    df["TotalCharges"] = df["TotalCharges"].str.strip().replace("", np.nan).astype(float)

    df = df.dropna()

        # 바이너리 변수에 대한 인코딩
    df["gender_encode"] = (df["gender"] == "Male").astype(int) #male -> 1 female -> 0
    df["Partner_encode"] = (df["Partner"] == "Yes").astype(int)
    df["Dependents_encode"] = (df["Dependents"] == "Yes").astype(int)
    df["PhoneService_encode"] = (df["PhoneService"] == "Yes").astype(int)
    df["PaperlessBilling_encode"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df = pd.get_dummies(df, columns=['Contract'], prefix=['Contract'])

    df["Contract_Month-to-month_encode"] = (df["Contract_Month-to-month"] == "True").astype(int)
    df["Contract_One year_encode"] = (df["Contract_One year"] == "True").astype(int)
    df["Contract_Two year_encode"] = (df["Contract_Two year"] == "True").astype(int)

    feature_names = df.select_dtypes(include="number").columns #숫자형 데이터 타입만을 포함하는 컬럼들을 가지고 새로운 feature 생성
    label_name = "Churn"

    X = df[feature_names]
    y = df[label_name]   

    split_count = int(df.shape[0] * 0.8)

    X_train = X[:split_count]
    y_train = y[:split_count]

    X_test = X[split_count:]
    y_test = y[split_count:]

    return X_train,y_train,X_test,y_test

    