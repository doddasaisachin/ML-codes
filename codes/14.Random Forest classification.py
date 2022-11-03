import pandas as pd
data=pd.read_csv("C:\\Users\\online\\Desktop\\codes\\HeartDisease.csv")
X =data.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12]].values
y =data.iloc[:,13].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier() 
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)
from sklearn import metrics
print("Classification Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)
import seaborn as sn
from matplotlib import pyplot as plt
plt.figure(figsize=(5,4))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
