from eda.eda_v1 import preprocess
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train_,y_train_,X_test_,y_test_ = preprocess()

def Svc(X_train,y_train,X_test,y_test):
    """svc 모델로 학습시키기"""
    svm_classifier = SVC()

    # 튜닝할 하이퍼파라미터 설정
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')

    # 데이터에 모델을 fitting
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터 출력
    print("최적의 하이퍼파라미터:", grid_search.best_params_)

    # 최적의 모델로 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("최적 모델의 정확도:", accuracy)


def logistic_regression(x_train,y_train,x_test,y_test):
    """Logistic Regression으로 학습시키기"""
    logreg_classifier = LogisticRegression()

    # 튜닝할 하이퍼파라미터 설정
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'max_iter': [50, 100, 200]
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(logreg_classifier, param_grid, cv=5, scoring='accuracy')

    # 데이터에 모델을 fitting
    grid_search.fit(x_train, y_train)

    # 최적의 하이퍼파라미터 출력
    print("최적의 하이퍼파라미터:", grid_search.best_params_)

    # 최적의 모델로 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("최적 모델의 정확도:", accuracy)



def Rf(X_train,y_train,X_test,y_test):
    """RandomForestClassifier 객체 생성"""
    rf_classifier = RandomForestClassifier()

    # 튜닝할 하이퍼파라미터 설정
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')

    # 데이터에 모델을 fitting
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼파라미터 출력
    print("최적의 하이퍼파라미터:", grid_search.best_params_)

    # 최적의 모델로 평가
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("최적 모델의 정확도:", accuracy)
def Knn(X_train,y_train,X_test,y_test):
    """KNN 객체 생성"""
    knn_classifier = KNeighborsClassifier()

    # 튜닝할 하이퍼 파라미터 설정
    param_grid = {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1은 맨해튼 거리, 2는 유클리디안 거리
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')

    # 데이터에 모델을 fitting
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼 파라미터 출력
    print("최적의 하이퍼 파라미터:", grid_search.best_params_)

    # 최적의 모델로 평가
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print("최적 모델의 정확도:", accuracy)
def Dt(X_train,y_train,X_test,y_test):
    """DecisionTreeClassifier 객체 생성"""
    dt_classifier = DecisionTreeClassifier()

    # 튜닝할 하이퍼 파라미터 설정
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy']
    }

    # GridSearchCV 객체 생성
    grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')

    # 데이터에 모델을 fitting
    grid_search.fit(X_train, y_train)

    # 최적의 하이퍼 파라미터 출력
    print("최적의 하이퍼 파라미터:", grid_search.best_params_)

    # 최적의 모델로 평가
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print("최적 모델의 정확도:", accuracy)



def main():
    dt(X_train_,y_train_,X_test_,y_test_)
    

if __name__ == "__main__":
    main()