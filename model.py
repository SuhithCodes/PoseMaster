import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import catboost as cb


# Function to train the stacking model
def train_stacking_model(X_train, y_train, selected_features):
    # Standardize features
    scaler = StandardScaler()
    X_train_selected_scaled = scaler.fit_transform(X_train.iloc[:, selected_features])
    
    # Base estimator (Random Forest)
    base_estimator_rf = RandomForestClassifier(n_estimators=5, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto')
    base_estimator_cb = cb.CatBoostClassifier(iterations=100, early_stopping_rounds=10, eval_metric='Accuracy')
    base_estimator_svc = SVC(kernel='poly', C=18, degree=2)
    
    # Stacking Classifier
    estimators = [
        ('svm', base_estimator_svc),
        ('catboost', base_estimator_cb),
        ('random_forest', base_estimator_rf)
    ]
    stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    stacking_model.fit(X_train_selected_scaled, y_train)
    return stacking_model

