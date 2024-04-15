import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

df=pd.read_excel('Lasso3.xlsx',sheet_name='Sheet1')
f=df.values
y=f[:,-1]
X=f[:,:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.fit_transform(x_train)
x_test_std=sc.transform(x_test)

from sklearn.svm import SVC
clf=SVC(probability=True,random_state=42).fit(x_train_std, y_train)

y_pred = clf.predict(x_test_std)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_value = auc(fpr, tpr)
y_scores = clf.predict_proba(x_test_std)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)
print("accuracy:", f'{accuracy:.4f}')
print("precision:", f'{precision:.4f}')
print("recall:", f'{recall:.4f}')
print("f1:", f'{f1:.4f}')
print("roc_auc:", f'{roc_auc:.4f}')

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()