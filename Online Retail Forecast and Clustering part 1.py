#!/usr/bin/env python
# coding: utf-8

# # PROJECT: ONLINE RETAIL FORECAST AND CLUSTERING

# ## OBJECTIVES
# 
# To develop a predictive model that classifies online retail transactions into specific categories and clusters. This will help uncover hidden patterns, improve customer segmentation,optimize inventory management and enhance sales forecasting

# ## PROJECT GOALS
#  1. Predictive Modelling: Build a model to classify transactions based on purchasing behavior.
#  2. Clustering Analysis: Identify customer and product segments using clustering techniques.
#  3. feature Engineering: create meaningful features from transactional data.
#  4. Sales Forecasting: Predict future sales trends based on historical data.
#  5. Customer segmentation: Group customers based on their purchasing behavior for targeted marketing

# ## DATA CLEANING AND PREPARATION
# Objective: Load the dataset, clean and preprocess it for anaylysis

# In[2]:


## Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## Load Dataset
df = pd.read_csv('Online_Retail.csv', encoding='ISO-8859-1')
df.head()


# In[3]:


## Description of our dataset
df.info()


# In[4]:


#Checking for duplictes
df.duplicated().sum()


# In[5]:


#Dropping Duplicates
df.drop_duplicates(inplace=True)


# In[6]:


## Check for missing values

df.isnull().sum()


# In[7]:


## Drop rows with missing values

df = df.dropna(subset =['CustomerID'])


# In[8]:


#Separate Negative Price
df = df[df["UnitPrice"] > 0 ]
df.head()


# In[9]:


#Renaming the Description to Products
df = df.rename(columns={"Description": "Product"})
df.head()


# In[10]:


invalid_stockcodes = ["POST", "DOT", "BANK CHARGES", "AMAZON FEES", "B",
                      "gift_0001_10", "gift_0001_20", "gift_0001_30", 
                      "gift_0001_40", "gift_0001_50"]

# Remove rows where StockCode is in the invalid list
df = df[~df["StockCode"].isin(invalid_stockcodes)]


# ## FEATURE ENGINEERING
# This involved the process of creating, transforming or selecting features (variables) from raw data to make it more suitable for analysis or machine learning models. In this case I extracted features like total sales, month, year, quarterky, weeks of the day and period of the day from invoicedate

# In[11]:


# Extracting useful Features

#Sales Feature
df['TotalSales'] = df['Quantity'] * df['UnitPrice']

# Convert 'InvoiceDate' to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])


# ##### Extracting Quarter

# In[12]:


# Extract Quarter from InvoiceDate

df['Quarter'] = df['InvoiceDate'].dt.quarter

#Group By Quarter and Sum of Sales
quarter_trends = df.groupby("Quarter")["TotalSales"].sum().reset_index()
quarter_trends = quarter_trends.sort_values(by = "Quarter")


# ##### Extracting Month

# In[13]:


# Extract Month Name
df['Month'] = df['InvoiceDate'].dt.strftime('%B') #strftime(string Format Time), %B(Python Magic Code to format date to Month).

# Define the correct month order
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Ensure 'Month' is a categorical variable with the correct order
df['Month'] = pd.Categorical(df['Month'], categories=month_order, ordered=True)

# Group by Month and Sum Sales
monthly_trends = df.groupby('Month')['TotalSales'].sum().reset_index()
monthly_trends = monthly_trends.sort_values(by = "Month")


# ##### Extracting Period of the Day

# In[14]:


# Define a function to categorize periods of the day
def get_period(hour):
    if hour < 12:
        return 'Morning'
    elif hour < 17:
        return 'Afternoon'
    elif hour < 20:
        return 'Evening'
    else:
        return 'Night'

# Apply the function to extract periods
df['Period'] = df['InvoiceDate'].dt.hour.apply(get_period)


# Define the correct order of periods
period_order = ["Morning", "Afternoon", "Evening", "Night"]

# Ensure 'Period' is a categorical variable with the correct order
df['Period'] = pd.Categorical(df['Period'], categories=period_order, ordered=True)

# Calculate total sales per period
period_trends = df.groupby('Period')['TotalSales'].sum().reset_index()
period_trends = period_trends.sort_values(by='Period')


# ##### Extracting Days of the Week

# In[15]:


#Extract the Days Of the Week
df['Days_of_Week'] = df['InvoiceDate'].dt.day_name()

# Define correct order of days
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Convert 'Days_of_Week' to categorical with correct order
df['Days_of_Week'] = pd.Categorical(df['Days_of_Week'], categories=days_order, ordered=True)

days_of_week_trends = df.groupby('Days_of_Week')['TotalSales'].sum().reset_index()
days_of_week_trends = days_of_week_trends.sort_values(by = "Days_of_Week")


# ## EXPLORATORY DATA ANALYSIS (EDA)
# The EDA is used for analyzing and summarizing a dataset to understand its patterns, detect anomalies and decide on the best preproecessing steps to machine learning.
# 
# I used the EDA to find patterns like Top 10 Best-selling produc and purchase trends liketd monthly Sales Tren, quarterly sales trend, daily sales trend and period of the day sales trendsds

# ## Top Selling Products

# In[16]:


#Top Selling Products

top_selling = df.groupby('Product')[['Quantity','TotalSales']].sum()


# In[17]:


# Set figure size
plt.figure(figsize=(8, 4))

# Create barplot
sns.barplot(x="TotalSales", y="Product", data=top_selling.head(10), color="green")

# Labelling the Visual
plt.xlabel("Total Sales",fontsize = 14)
plt.ylabel("Products",fontsize = 14)
plt.title("Top 10 Best-Selling Products", fontsize=14)
plt.grid(axis="x", linestyle="--")

# Show the plot
plt.show()


# ### Interpretation and Business Insight
# 
# The bar chart shows the Top 10 Best Selling Products based on total sales. The chart revealed that SET 2 TEA TOWELS I LOVE LONDON and SPACEBOY BABY GIFT SET are the highest selling products, both of them exceeding 7,000 sales. Meanwhile some products like 4 PURPLE FLOCK DINNER CANDLes have singnficantly lower sales
# 
# Business Insight:
# 
# Businesses should stock more of the high demand products like SET 2 TEA TOWELS I LOVE LONDON and SPACEBOY BABY GIFT SET while reconsidering the supply of low-performing items
# 

# ### Purchase Trend

# #### Monthly Sales Trend

# In[18]:


# Create the plot
plt.figure(figsize=(8, 4))
sns.lineplot(data=monthly_trends, x='Month', y='TotalSales', marker='o', linewidth=2, color='red')

# Customize the chart
plt.title('Monthly Sales Trend', fontsize=14)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels if needed
plt.grid(True)  # Add grid for better readability

# Show the plot
plt.show()


# ### Interpretation and Business Insight
# 
# The chart reveals that November has the highest sales. while still high, December sees a slight drop compared to November. 
# 
# September to November also show an upward trend, possibly due to back-to-school or pre-holiday sales.
# 
# The overall trend suggests a gradual increase from mid-year, peaking in December.
# 
# 
# Business Insight:
# 
# 1. The Business should increase inventory before peak sales months (September- November) and reduce stock in slower months like April and July
# 2. Promotions should bbe ramped up in low sales month to smooth revenue  streams

# ### Quarter of the Year Sales Trend

# In[19]:


#Create the plot

plt.figure(figsize=(8, 4))

sns.lineplot(data=quarter_trends, x='Quarter', y='TotalSales', marker='o', linewidth=2, color='blue')

# Customize the chart
plt.title('Quarter Of The Year Sales Trend', fontsize=14)
plt.xlabel('Quarter Of the Year', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=360)  
plt.grid(True)  

# Show the plot
plt.show()


# ### Interpretation and Business Insight
# 
# The chart revealed that Q4 (Oct-Dec) has the highest sales, confirming what I saw in the monthly trend chart (November peak).
# 
# Each quarter shows can increase in sales,meaning business performance improves over time.
# 
# Q1 has the lowest sales, which is common because consumers often reduce spending after the holidays
# 
# Business Insight:
# 
# The business should invest more in Q4 campaigns since they generate the highest returns.

# ### Daily Sales Trend

# In[20]:


#Create the plot

plt.figure(figsize=(8, 4))

sns.lineplot(data=days_of_week_trends, x='Days_of_Week', y='TotalSales', marker='o', linewidth=2, color='orange')

# Customize the chart
plt.title('Days of the Week Sales Trend', fontsize=14)
plt.xlabel('Days Of the Week', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=360)  
plt.grid(True)  

# Show the plot
plt.show()


# ### Intepretation and Business Insight
# 
# Thursday has the highest sales, suggesting its the peak shopping day. It could be due to special promotions, payday shopping or weekly dicounts
# 
# The sales are relatively stable from Monday to Wednesday but increae significantly on Thursday.
# 
# Sales drop sharply on Friday and hit the lowest point on Saturday.
# 
# Business Insight:
# 
# The business should boost marketing campaigns on Thursdays to maximize peak sales.
# 
# The business should consider weekend deals to drive engagement on Saturday and Sunday

# ### Period of the Day Sales Trend

# In[21]:


#Create the plot

plt.figure(figsize=(8, 4))

sns.lineplot(data=period_trends, x="Period",y="TotalSales",marker="o",linewidth=2, color= "purple")

# Customize the chart
plt.title('Period of the Day Trend', fontsize=14)
plt.xlabel('Period Of The Day', fontsize=14)
plt.ylabel('Total Sales', fontsize=14)
plt.xticks(rotation=360)  
plt.grid(True)  

# Show the plot
plt.show()


# ### Intepretation and Business Insight
# 
# The chart reveales that Sales peak in the Afternnon indicating that most transactions occur during this period.
# 
# Morning sales are moderate, showing steady activity but lower than the afternoon peak.
# 
# Sales decline sharply in the evening and reach the lowest point at night. The drop might be that people might be engaged in other activities such as socializing or relaxing while the low night sales might be due to rest of fewer promotional campaigns.
# 
# Business Insights:
# 
# Boost afternoon marketing efforts to capitalize on peak sales.

# ## CLUSTERING AND SEGMENTATION
#  

# ### Customer Segmentation
# 
# To effectively work on the clustering and customer segmentation i decided to use Recency, Frequency and Monetary Value (RFM) method as the basis for that in order to understand customer behavior and grouping customers into meaningful segments. 

# ### Recency(R)
# 
# This reveals how recently a customer made a purchase

# In[22]:


#Setting the reference date to the end date of the analysis
reference_date = df["InvoiceDate"].max()

#Grouping data by customer and getting the latest purchase date for each customer
grouped = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()

#Calculating the recency for each customer by subtracting the most recent purchase date from the reference date
grouped["Recency"] = (reference_date - grouped["InvoiceDate"]).dt.days

#Merging the recency value to the dataframe
df = df.merge(grouped[["CustomerID", "Recency"]], on = "CustomerID", how = "left")


# In[23]:


df


# ### Frequency(F)
# 
# This tells how often a customer makes a purchase

# In[24]:


#Calculating the frequency for each customer by counting the number of purchases
frequency = df.groupby("CustomerID")["InvoiceNo"].count()

#Merging the frequency value to the dataframe
df = df.merge(frequency, on = "CustomerID", how = "left")
df.rename(columns={"InvoiceNo_x" : "InvoiceNo", "InvoiceNo_y" : "Frequency"}, inplace = True)


# In[25]:


df


# ### Monetary Value
# 
# This shows how much money a customer spends

# In[26]:


# Step 1: Compute Monetary value per customer (sum of TotalSales)
monetary_value = df.groupby("CustomerID")["TotalSales"].sum().reset_index()

# Step 2: Merge the monetary value with the main dataframe
df = df.merge(monetary_value, on = "CustomerID", how = "left")

# Rename the column for clarity
df.rename(columns={"TotalSales_x" : "TotalSales", "TotalSales_y" : "MonetaryValue"}, inplace = True)


# ### Recency, Frequency and Monetary Value (RFM)

# In[28]:


# Ensure unique customers before clustering
clustering_features = df.groupby("CustomerID").agg({
    "Recency": "min",          # Most recent purchase (smaller is better)
    "Frequency": "max",        # Total number of transactions
    "MonetaryValue" : "max"
}).reset_index()

clustering_features


# #### Determining the Optimal Number of Clusters
# 
# I used the Elbow method to determine the optimal number of clusters for K-means. Doing this is critical because it helps find the best number of groups (clusters) that the data should be divided into such that data points within the same cluster are as smilar as possible and also data points in different clusters are as disimilar as possible

# In[29]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardizing the features
scaler = StandardScaler()
clustering_features_scaled = scaler.fit_transform(clustering_features[["Recency", "Frequency", "MonetaryValue"]])

# Finding the optimal number of clusters using the Elbow Method
inertia = []
K_range = range(1, 11)  # Checking for clusters from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(clustering_features_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
sns.lineplot(x=K_range, y=inertia, marker="o", color="b")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method for Optimal K")
plt.show()


# In[30]:


clustering_features = clustering_features.copy()


# #### Applying K-means Clustering

# In[33]:


# Apply K-Means clustering with K=4
kmeans = KMeans(n_clusters = 3, random_state = 42, n_init = 10)
clustering_features.loc[:, "CustomerSegment"] = kmeans.fit_predict(clustering_features_scaled)

# Display the first few rows with assigned clusters
clustering_features


# #### Analyse and intepret Cluster

# In[ ]:


# Merge CustomerSegment back into the original df using CustomerID
df = df.merge(clustering_features[["CustomerID", "CustomerSegment"]], on = "CustomerID", how = "left")


# ## PREDICTIVE MODELLING
# 
# The aim of this is to build a machine learning model to classify transactions

# In[35]:


df.columns


# In[36]:


df1 = df.drop(["InvoiceNo", "StockCode", "Product", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"], axis = 1)
df1


# #### *Reasons for the drop:
# Dropping these columns remove irrelevant or non-predictive data, reducing noise or overfitting. Retaining features like country, total sales and others captures meaningful patterns and trends for predictive modeling.

# ### Handling Outliers
# 
# I handled the outliers in order not to allow them skew my analysis, distort model performance and lead to incorrect predictions

# In[65]:


sns.boxplot(df, x = "Recency")
plt.show()


# In[37]:


## Function to calculate lower and upper whiskers
def calculate_whiskers(col):
    q1, q3 = np.percentile(col, [25, 75])
    iqr = q3 - q1
    lw = q1 - 1.5 * iqr
    uw = q3 + 1.5 * iqr
    return lw, uw


# In[38]:


num_col = [col for col in df1.select_dtypes(include=['int64', 'float64']).columns if col != "CustomerSegment"]
num_col


# In[39]:


## Outlier Handling using winsorization
for i in num_col:
    lw, uw = calculate_whiskers(df1[i])
    df1[i] = np.where(df1[i] < lw, lw, df1[i])
    df1[i] = np.where(df1[i] > uw, uw, df1[i])


# ### Feature Encoding
# This feature encoding helps in converting catagorical data into numerical data, hence I decided to employ it in other to convert my numerical data like Month and Days of the week into categorical data seeing that most algorithms  require numerical input and cannot directly work with categorical data

# In[40]:


from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# Define the order for 'Month' and 'Days_of_Week'
month_order = [
    "January", "February", "March", "April", "May", "June", 
    "July", "August", "September", "October", "November", "December"
]
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Apply Ordinal Encoding
ordinal_encoder = OrdinalEncoder(categories=[month_order, days_order])
# Initialize LabelEncoders
label_encoder = LabelEncoder()

lbe = ["Country", "Period"]

for col in lbe:
    df1[col] = label_encoder.fit_transform(df1[col])

df1[["Month", "Days_of_Week"]] = ordinal_encoder.fit_transform(df1[["Month", "Days_of_Week"]])


# In[66]:


df1


# ### Handling Imbalance
# 
# Handling imbalance is necessary in order to prevent a situation where the model favors the majority class, ignoring the minority as seen in the customer segment

# In[41]:


sns.countplot(df1, x = "CustomerSegment")
plt.show()


# In[42]:


from sklearn.utils import resample

# Separate classes
df_1 = df1[df1["CustomerSegment"] == 0]  # Minority class
df_2 = df1[df1["CustomerSegment"] == 1]  # Majority class
df_3 = df1[df1["CustomerSegment"] == 2]  # Minority class

min_size = max(len(df_1), len(df_2), len(df_3))  # Get largest class size

df_1_resampled = resample(df_1, replace=True, n_samples=min_size, random_state=42)
df_2_resampled = resample(df_2, replace=True, n_samples=min_size, random_state=42)
df_3_resampled = resample(df_3, replace=True, n_samples=min_size, random_state=42)

df_balanced = pd.concat([df_1_resampled, df_2_resampled, df_3_resampled]).sample(frac=1, random_state=42)

df_balanced["CustomerSegment"].value_counts()


# ### Feature Scaling

# In[43]:


features = df1.drop("CustomerSegment", axis = 1)
target = df1["CustomerSegment"]


# In[44]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 1)


# In[45]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sc_X_train = scaler.fit_transform(X_train)
sc_X_test = scaler.transform(X_test)


# ## Model Selection
# 
# The aim of this is to choose the best statistical or machine learning model for this dataset

# In[46]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report, confusion_matrix


# In[47]:


## Define models

models = {
    "lr" : LogisticRegression(),
    "rfr" : RandomForestClassifier(),
    "xg" : XGBClassifier(),
    "knn" : KNeighborsClassifier(),
    "dt" : DecisionTreeClassifier()
}

## Train and evaluate each model
for name, model in models.items():
    model.fit(sc_X_train, y_train)
    y_pred = model.predict(sc_X_test)

    ## Evaluation
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")


    print(f"Model: {name}")
    print(f"F1_Score: {f1:.2f}")
    print(f"Recall_Score: {recall:.2f}")


# Based on the F1-score and Recall-score displayed for the different models, the best model would ideally be the one with the higheest scores Random Forest, XGBoost and Decision Tree. nevertheless i choose XGboost (xg) because it tends to generalise better compared to others and also handles large datasets well and reduces overfitting using boosting

# ### FEATURE IMPORTANCE ANALYSIS  
# 
# This involved choosing the most relevant importantt features (input variables) from my dataset to improve my machine learning model's performance

# In[48]:


features.shape


# In[49]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif
f1_list = []
accuracy_list = []

xg = XGBClassifier()

for i in range(1, 11):
  selector = SelectKBest(mutual_info_classif, k = i)
  selector.fit(sc_X_train, y_train)

  sel_X_train = selector.transform(sc_X_train)
  sel_X_test = selector.transform(sc_X_test)

  xg.fit(sel_X_train, y_train)
  kbest_test = xg.predict(sel_X_test)

  f1_test = round(f1_score(y_test, kbest_test, average = "weighted"), 2)
  accuracy_test = round(accuracy_score(y_test, kbest_test), 2)

  f1_list.append(f1_test)
  accuracy_list.append(accuracy_test)


# In[50]:


fig, ax = plt.subplots(figsize = (20, 10))

x = np.arange(1, 11)
y = f1_list

ax.bar(x, y, width = 0.2)
ax.set_xlabel("Number of features selected using mutual information")
ax.set_ylabel("F1 Score (weighted)")
ax.set_ylim(0, 1.2)
ax.set_xticks(np.arange(1, 11))
ax.set_xticklabels(np.arange(1, 11), fontsize = 12)

for i, v in enumerate(y):
  plt.text(x = i + 1, y = v + 0.05, s = str(v), ha = "center")

plt.tight_layout()


# In[51]:


selector = SelectKBest(mutual_info_classif, k = 8)
selector.fit(sc_X_train, y_train)

selected_features_mask = selector.get_support()
selected_features = X_train.columns[selected_features_mask]


# In[52]:


selected_features

