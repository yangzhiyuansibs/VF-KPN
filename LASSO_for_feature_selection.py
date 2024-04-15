import pandas as pd
import numpy as np
df=pd.read_excel('三.xlsx',sheet_name='Sheet1')
f=df.values
y=f[:,0]
X=f[:,1:]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

alphas=np.logspace(-5, 2, 200)
cv_scores = []

from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
for alpha in alphas:
    lasso = Lasso(alpha=alpha, random_state=42)
    scores = cross_val_score(lasso,X_scaled,y,cv=5,scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())
best_alpha=alphas[np.argmin(cv_scores)]
print(f"最佳alpha值: {best_alpha:.2f}")

a=float(f"{best_alpha:.2f}")
print(a)
lasso=Lasso(alpha=a, random_state=42)
lasso.fit(X_scaled,y)

non_zero_indices=np.nonzero(lasso.coef_)[0]
selected_features=X[:,non_zero_indices]

selected_features_df = pd.DataFrame(data=selected_features, columns=df.columns[1:][non_zero_indices])
selected_features_df['Target']=y
selected_features_df.to_excel('3.xlsx', index=False)
