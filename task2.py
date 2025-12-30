import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
data =pd.read_csv("task2.csv")
#print(data.head())
data.drop("card",axis =1,inplace =True,errors="ignore")
# print(data.head())
def fill_card(card):
 if pd.isna(card):
        return 0
 elif card =="cash":
     return 1
 else:
     return 2
 
data["cash_type"]=data["cash_type"].apply(fill_card)
print(data.head())
data["date"]=pd.to_datetime(data["date"])
data["YEAR"]=data["date"].dt.year
data["month"]=data["date"].dt.month
data.drop("date",axis =1,inplace =True,errors ="ignore")
data["datetime"]=pd.to_datetime(data["datetime"])
data["day"]=data["datetime"].dt.day
data["dayofweek"]=data["datetime"].dt.dayofweek
data["hour"]=data["datetime"].dt.hour
data["minute"] =data["datetime"].dt.minute
data["second"]=data["datetime"].dt.second
data.drop("datetime",axis =1,inplace =True,errors ="ignore")
# print(data.head())
# print(data.info())
salesby_coffee= data.groupby("coffee_name")["money"].sum().reset_index()
salesby_coffee.columns= ["coffee_name","coffee_sales"]
print(salesby_coffee.head())
pd.concat([salesby_coffee,data],axis =1)
data =data.merge(salesby_coffee,on ="coffee_name")
# print(data.head())
sns.countplot(data["coffee_name"])
sns.displot(data["month"],bins =36,kde =False)
sns.boxplot(x ="month",y ="coffee_sales",data =data)
sns.scatterplot(x ="month",y ="coffee_sales",hue ="cash_type",data =data)
sns.scatterplot(x ="hour",y ="coffee_sales",hue ="cash_type",data =data)
sns.countplot(data["dayofweek"])
sns.scatterplot(x ="dayofweek",y ="coffee_sales",hue ="cash_type",data =data)
data.drop("money",axis =1,inplace =True,errors ="ignore")
# print(data.head())
salesby_minute= data.groupby("minute")["coffee_sales"].sum().reset_index()
salesby_minute.columns= ["minute","minute_sales"]
data =data.merge(salesby_minute,on ="minute")
salesby_coffee =data["coffee_name"].value_counts().reset_index()
salesby_coffee.columns =["coffee_name","coffee_count"]
data =data.merge(salesby_coffee,on ="coffee_name")
# print(data.head())
numeric_data =data.select_dtypes(include =[np.number])
print("\nCorrelation with coffee_sales:")
print(numeric_data.corr()["coffee_sales"].sort_values())
# sns.heatmap(numeric_data.corr(),annot =True,cmap ="coolwarm")
sns.countplot(data["minute"])
plt.show()
X =data.drop("coffee_sales",axis =1)
y =data["coffee_sales"]
Xtrain, Xtest, ytrain, ytest =train_test_split(X,y,test_size =0.2,random_state =42)
categorical_features = ['coffee_name']
numeric_features = ['cash_type', 'YEAR', 'month', 'day', 'dayofweek', 'hour', 'minute', 'second']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),  # Scale numerics
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)  # One-hot strings
    ])

# Step 4: Full Pipeline (preprocess + model)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Step 5: Train the model
pipeline.fit(Xtrain, ytrain)

# Step 6: Predict and Evaluate
y_pred = pipeline.predict(Xtest)
mse = mean_squared_error(ytest, y_pred)
r2 = r2_score(ytest, y_pred)

print(f"Model trained successfully!")
print(f"Test MSE: {mse:.2f}")
print(f"Test R²: {r2:.2f}")
new_data = pd.DataFrame({
    'cash_type': [2],
    'coffee_name': ['Americano'],
    'YEAR': [2024],
    'month': [3],
    'day': [1],
    'dayofweek': [4],
    'hour': [10],
    'minute': [15],
    'second': [50]
})
predicted_money = pipeline.predict(new_data)[0]
print(f"Predicted money for new Latte order: ${predicted_money:.2f}")
sns.scatterplot(x=ytest, y=y_pred)
plt.show()



