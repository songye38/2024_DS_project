"""각기 다른 버전의 eda 모듈"""
################################
#Contract_month_to_month
#onLine_security
#tech_suppport
#internet_service_fiber_optic
################################


import pandas as pd
import numpy as np

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


    #상관관계가 높은 범주형 변수들 변환해주기
    df = pd.get_dummies(df, columns=['Contract'], prefix=['Contract'])
    df = pd.get_dummies(df, columns=['OnlineSecurity'], prefix=['OnlineSecurity'])
    df = pd.get_dummies(df, columns=['TechSupport'], prefix=['TechSupport'])
    df = pd.get_dummies(df, columns=['InternetService'], prefix=['InternetService'])

    #위에서 바꾼 변수들을 다시 숫자형으로 바꿔주기
    df["Contract_Month-to-month"] = (df["Contract_Month-to-month"] == "True").astype(int)
    df["OnlineSecurity_No"] = (df["OnlineSecurity_No"] == "True").astype(int)
    df["TechSupport_No"] = (df["TechSupport_No"] == "True").astype(int)
    df["InternetService_Fiber optic"] = (df["InternetService_Fiber optic"] == "True").astype(int)


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

    