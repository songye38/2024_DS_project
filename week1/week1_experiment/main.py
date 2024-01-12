from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import neptune
import neptune.integrations.sklearn as npt_utils


from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("NEPTUNE_API_KEY")

run = neptune.init_run(
    project="songye/IBM-customer-data",
    api_token=api_key,
)  # your credentials

parameters = {
    "n_estimators": 90,
    "learning_rate": 0.07,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
}
run["parameters"] = parameters

gbc = GradientBoostingClassifier(**parameters)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=28743
)
gbc.fit(X_train, y_train)

# Neptune integration with scikit-learn works with
# the regression and classification problems as well.
# Check the user guide in the documentation for more details:
# https://docs.neptune.ai/integrations/sklearn
run["classifier"] = npt_utils.create_classifier_summary(
    gbc, X_train, X_test, y_train, y_test
)

run.stop()