import pandas as pd
#from ydata_profiling import ProfileReport
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv("kidney_disease_dataset.csv")

#profile = ProfileReport(data, title="Kidney Disease", explorative=True)
#rofile.to_file("kidney.html")

target = "Dialysis_Needed"
x = data.drop(columns=[target])
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=31)

smote = SMOTE(sampling_strategy=0.2, random_state=31)
x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

params = {
    "n_estimators": [50, 100, 150, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [5, 10, 15, 20],
}

model = GridSearchCV(
    estimator = RandomForestClassifier(class_weight="balanced", random_state=31),
    param_grid = params,
    scoring = "precision",
    cv=6,
    verbose=2,
    n_jobs=-1
)

model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print(model.best_params_)
print(model.best_score_)

print(classification_report(y_test, y_predict))