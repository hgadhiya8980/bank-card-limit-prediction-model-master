from flask import Flask, render_template, request

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("credit_predicated_file.csv")

X=df.drop("Credit_Limit", axis=1)
y=df["Credit_Limit"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=123)

sc = StandardScaler()
sc.fit(X_train,y_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = joblib.load("card_limit_prediction_model.pkl")

def predict_card_limit(model,Attrition_Flag, Customer_Age, Gender, Dependent_count,
                       Education_Level, Marital_Status, Income_Category, Card_Category,
                       Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
                       Contacts_Count_12_mon, Total_Revolving_Bal,
                       Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
                       Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio):
    x=np.zeros(len(X.columns))
    
    x[0]=Customer_Age
    x[1]=Dependent_count
    x[2]=Months_on_book
    x[3]=Total_Relationship_Count
    x[4]=Months_Inactive_12_mon
    x[5]=Contacts_Count_12_mon
    x[6]=Total_Revolving_Bal
    x[7]=Avg_Open_To_Buy
    x[8]=Total_Amt_Chng_Q4_Q1
    x[9]=Total_Trans_Amt
    x[10]=Total_Trans_Ct
    x[11]=Total_Ct_Chng_Q4_Q1
    x[12]=Avg_Utilization_Ratio
    
    if "Attrition_Flag_"+Attrition_Flag in X.columns:
        flag_count = np.where(X.columns == "Attrition_Flag_"+Attrition_Flag)[0][0]
        x[flag_count]=1
        
    if "Gender_"+Gender in X.columns:
        gender_count = np.where(X.columns == "Gender_"+Gender)[0][0]
        x[gender_count]=1
    
    if "Education_Level_"+Education_Level in X.columns:
        edu_count = np.where(X.columns == "Education_Level_"+Education_Level)[0][0]
        x[edu_count]=1
        
    if "Marital_Status_"+Marital_Status in X.columns:
        mrg_count = np.where(X.columns == "Marital_Status_"+Marital_Status)[0][0]
        x[mrg_count]=1
        
    if "Income_Category_"+Income_Category in X.columns:
        income_count = np.where(X.columns == "Income_Category_"+Income_Category)[0][0]
        x[income_count]=1
        
    if "Card_Category_"+Card_Category in X.columns:
        card_count = np.where(X.columns == "Card_Category_"+Card_Category)[0][0]
        x[card_count]=1
        
    x = sc.transform([x])[0]
    
    return model.predict([x])[0]

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    Attrition_Flag = request.form["Attrition_Flag"]
    Customer_Age = request.form["Customer_Age"]
    Gender = request.form["Gender"]
    Dependent_count = request.form["Dependent_count"]
    Education_Level = request.form["Education_Level"]
    Marital_Status = request.form["Marital_Status"]
    Income_Category = request.form["Income_Category"]
    Card_Category = request.form["Card_Category"]
    Months_on_book = request.form["Months_on_book"]
    Total_Relationship_Count = request.form["Total_Relationship_Count"]
    Months_Inactive_12_mon = request.form["Months_Inactive_12_mon"]
    Contacts_Count_12_mon = request.form["Contacts_Count_12_mon"]
    Total_Revolving_Bal = request.form["Total_Revolving_Bal"]
    Avg_Open_To_Buy = request.form["Avg_Open_To_Buy"]
    Total_Amt_Chng_Q4_Q1 = request.form["Total_Amt_Chng_Q4_Q1"]
    Total_Trans_Amt = request.form["Total_Trans_Amt"]
    Total_Trans_Ct = request.form["Total_Trans_Ct"]
    Total_Ct_Chng_Q4_Q1 = request.form["Total_Ct_Chng_Q4_Q1"]
    Avg_Utilization_Ratio = request.form["Avg_Utilization_Ratio"]
        
    

    predicated_price = predict_card_limit(model,Attrition_Flag, Customer_Age, Gender, Dependent_count,
                       Education_Level, Marital_Status, Income_Category, Card_Category,
                       Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
                       Contacts_Count_12_mon, Total_Revolving_Bal,
                       Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
                       Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio)
    
    
    
    return render_template("index.html", prediction_text="predicated value of card limit {}".format(predicated_price))
    

if __name__ == "__main__":
    app.run()
    
    
    
    
    
    
    
    
    
    
    
    
    