from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from setup import iris
from sklearn import metrics

#Get rid of target and species
X = iris.drop(['target', 'species'], axis=1)

X = X.to_numpy()[:, (2,3)]
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

training_prediction = log_reg.predict(X_train)
print(training_prediction)

test_prediction = log_reg.predict(X_test)
print(test_prediction)

#Results & data
#print(metrics.classification_report(y_train, training_prediction, digits=3))
#print(metrics.confusion_matrix(y_train, training_prediction))

#More Results
print(metrics.classification_report(y_test, test_prediction, digits=3))
print(metrics.confusion_matrix(y_test, test_prediction))