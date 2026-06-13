import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn .ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
df=pd.read_csv("cardekho.csv",encoding="latin1")
df["cardekho"]=2025-df['year']
df.drop("year",axis=1,inplace=True)
df=pd.get_dummies(df,drop_first=True)
X=df.drop("selling_price",axis=1)
y=df["selling_price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mae=mean_absolute_error=(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
r2=r2_score(y_test,y_pred)
print("MAE:",mae)
print("RMSE:",rmse)
print("R2 Score:",r2)
sample=X_test.iloc[[0]]
predicted_price=model.predict(sample)
print("Predicted car price:",predicted_price[0])
import joblib
joblib.dump(model,"cardekho_model.pkl")
import os
print(os.listdir())

