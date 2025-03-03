# ONLINE RETAAIL FORECAST AND CLUSTERING

## MY ROLE 

1.	Predictive Modeling: Build a model to classify transactions based on purchasing behavior.
2.	Clustering Analysis: Identify customer and product segments using clustering techniques.
3.	Feature Engineering: Create meaningful features from transactional data.
4.	Sales Forecasting: Predict future sales trends based on historical data.
5.	Customer Segmentation: Group customers based on their purchasing behavior for targeted marketing.

## PROJECT METHODOLOGY: 
1. Data Collection: I Gathered transactional data from an online retail platform and ensured data includes details like customer ID, product ID, purchase date, quantity, price, and payment method.
2. Data Preprocessing: I handled missing values and outliers, converted date-time features into meaningful formats and standardized or normalize numerical features if needed.
3. Exploratory Data Analysis (EDA): I generated summary statistics (mean, median, mode, etc.) and Visualized sales trends, top-selling products, and customer purchase behavior.
5. Feature Engineering: I created new features like purchase frequency, average spend, and recency of last purchase and Categorized products and customers based on spending patterns.
6. Predictive Classification Model: I trained a machine learning model to classify transactions based on purchasing behavior and evaluated model performance using accuracy, precision, recall, and F1-score.
7. Clustering Analysis: I Applied clustering techniques (e.g., K-Means) to segment customers and products and also Visualized interpreted clusters to identify purchasing patterns.
8. Sales Forecasting: I used Used time series models (Prophet) to predict future sales trends and also Evaluated forecast accuracy with metrics like RMSE and MAPE.
9. Customer Segmentation: I grouped customers based on their purchasing habits (e.g., Low purchase customers, Moderate purchase customers and high purchase customers) 

## TOOLS USED

Excel, Python, Matplolib, Seabon, K-Means Model. 


## ANALYSIS AND INSIGHTS

### Exploratory Data Analysis (EDA): 

Summary statistics and trend analysis

The bar chart below shows the Top 10 Best Selling Products based on total sales. The chart revealed that SET 2 TEA TOWELS I LOVE LONDON and SPACEBOY BABY GIFT SET are the highest selling products, both of them exceeding 7,000 sales. Meanwhile some products like 4 PURPLE FLOCK DINNER CANDLes have singnficantly lower sales

![image](https://github.com/user-attachments/assets/f4986ffc-7b9d-43e5-8a9a-20dfd50d1eec)
  

The chart revealed that Q4 (Oct-Dec) has the highest sales, confirming what we saw in the monthly trend chart (November peak).

Each quarter shows can increase in sales, meaning business performance improves over time.

Q1 has the lowest sales, which is common because consumers often reduce spending after the holidays

![image](https://github.com/user-attachments/assets/30d5b810-f842-4b70-bb3b-e9b096dd2d79)

 
The chart reveals that November has the highest sales. while still high, December sees a slight drop compared to November. 

September to November also show an upward trend, possibly due to back-to-school or pre-holiday sales.

The overall trend suggests a gradual increase from mid-year, peaking in December.

![image](https://github.com/user-attachments/assets/361d7f07-dc57-42a3-99d8-8f20c706a290)

 
Thursday has the highest sales, suggesting its the peak shopping day. It could be due to special promotions, payday shopping or weekly discounts

The sales are relatively stable from Monday to Wednesday but increase significantly on Thursday.

Sales drop sharply on Friday and hit the lowest point on Saturday.

![image](https://github.com/user-attachments/assets/67ac46e4-8f98-4652-97de-40c23a2c199a)




 
The chart below reveals that Sales peak in the Afternoon indicating that most transactions occur during this period.

Morning sales are moderate, showing steady activity but lower than the afternoon peak.

Sales decline sharply in the evening and reach the lowest point at night. The drop might be that people might be engaged in other activities such as socializing or relaxing while the low night sales might be due to rest of fewer promotional campaigns

![image](https://github.com/user-attachments/assets/d7f768ad-756d-464d-b285-c5b22f0b033d)

### Clustering and Customer Segmentation 
 
The chart below reveals how customers behave based on Recency (how long ago they made a purchase) and Frequency (how often they buy).
Customer Segment

i.	Green (High purchase): These are customers who buy frequently and tend to make purchases recently.

ii.	Orange (Moderate Purchase): They buy occasionally but are still engaged

iii.	Blue (Low Purchase): These customers haven’t bought in a long time and also don’t purchase often.

Based on the chart, Customers who buy often tend to come back quickly (green cluster). Some customers buy occasionally and have moderate recency (Orange cluster) while many customers haven't bought in a long time and don’t buy often (Blue cluster)

![image](https://github.com/user-attachments/assets/4e044c48-109e-400d-8e1a-cbade1c47d05)

### Predictive Classification Mode 

Machine learning model selection
  
Based on the F1-score and Recall-score displayed for the different models, the best model would ideally be the one with the highest scores Random Forest, XGBoost and Decision Tree. nevertheless I choose XGboost (xg) because it tends to generalize better compared to others and also handles large datasets well and reduces overfitting using boosting.

### Feature Importance Analysis  
 
The chart below show which factors (features) are most important when predicting customer behavior using XGBoost model. The Longer the bar, the more important the feature.

The chart revealed that Monetary Value (Most important Feature) i.e how much money a customer spends plays a biggest role in predictions. High spending customers are more valuable, so this feature helps in customer segmentation.

Recency:
This refers to how recently a customer made a purchase. if a customer has not bough anything in a long time, they might be at risj of leaving.

Frequency:
This means how often a customer buys is also an important  factor but not as strong as monetary value or recency

Oher Features:
Factors like Year, Month, country and sales periods still matter but dont contribute as much as to predictions.

![image](https://github.com/user-attachments/assets/77b017d8-3101-445f-aed8-30cf6f1aab73)

### Sales Forecasting Model, Time Series Forecasting and Business Insights on Future Sales Trends

The time series looks at the past data over time to find patterns. it helps understand how sales have changed historically and whether there are recurring patterns. The Sales Forecasting uses time series analysis (along with machine learning models) to predict future sales.

The Blue Line in the chart below represents past sales data (actual sales after the model made predictions). 

The red line represents future sales data (actual sales after the model made predictions)

The green shaded area shows the model's confidence interval (how sure it is about its predictions)

The black shaded line separates past data (training) from future data (testing)

The charts show that, the online store had fluctuating sales with some months higher than other.

The model predicted sales to stay within a stable range (as seen in the confidence level) but actual sales spiked unexpectedly (likely due to external factors like a holiday season, discounts, viral trends or a special promotion.

![Uploading image.png…]()


### Evaluation of forecast accuracy 
 
The Mean Absolute Error (MAE) measures the average different between the actual sales values and the predicted values by the mode. A MAE of 22,613.51 means that, on average, the model sales predictions differ from the actual sales by about 22,613 units.

The model accuracy in prediction depends on the scale of the data:

if total sales are in million, an error of 22,613 might not be significant. but if the total sales are much smaller, this error is quite large, meaning the models prediction are not very accurate.

Thus looking at the Total Sales of the business in the past years, 2010 amounted a total sale of 776,240.91 and in 2011 it increased significantly to a total sales of 9,478,095.

Relative to 2010 Sales, an average error of 22,613 is roughly 3% of 776K while relative to 2011 sales (9.48M), an average error of 22,613 is about 0.24% of 9.48M. 

Hence we can conclude that on an annual scale, the model seems reasonably accurate because 22K is a small fraction compared to millions in total sales.



## MAJOR KEY FINDINGS

1. The chart revealed that SET 2 TEA TOWELS I LOVE LONDON and SPACEBOY BABY GIFT SET are the highest selling products, both of them exceeding 7,000 sales. Meanwhile some products like 4 PURPLE FLOCK DINNER CANDLes have significantly lower sales
2. The chart revealed that Q4 (Oct-Dec) has the highest sales with Q1 having the lowest sales.
3. November has the highest sales. The overall trend suggests a gradual increase from mid-year, peaking in December
4. Thursday has the highest sales, The sales are relatively stable from Monday to Wednesday but increase significantly on Thursday.
5. The Sales are peak in the Afternoon. Morning sales are moderate, showing steady activity but lower than the afternoon peak. Sales decline sharply in the evening and reach the lowest point at night.
6. There is low population of Customers who buy often that tend to come back quickly (green cluster). Some customers buy occasionally and have moderate recency (Orange cluster) while customers that haven't bought in a long time and don’t buy often (Blue cluster) are high in population.
7. To improve the effectiveness of the models when it comes to prediction, more features like should be recorded like holiday season, discounts, viral trends or a special promotion.



## RECOMMENDATIONS FOR TARGETED MARKETING STRATEGIES
1. The Businesse should stock more of the high demand products like SET 2 TEA TOWELS I LOVE LONDON and SPACEBOY BABY GIFT SET while reconsidering the supply of low-performing items
2. The Business should increase inventory before peak sales months (September- November) and reduce stock in slower months like April and July
3. Promotions should be ramped up in low sales month to smooth revenue streams
4. The business should invest more in Q4 campaigns since they generate the highest returns.
5. The business should boost marketing campaigns on Thursdays to maximize peak sales.
6. The business should consider weekend deals to drive engagement on Saturday and Sunday
7. To retain high purchase customers (Green Clusters). The Business can implement a loyalty program or exclusive discounts to reward them.
8. For the moderate-purchase customers (Orange clusters), the business can used personalized promotions or email reminders to increase engagement.
9. Lastly for the Low Purchase Customers (Blue segment), they need to be reactivated. The business can implement re-engagement campaigns with special offers or discounts, use targeted ads or emails to remind them of products they previously showed interest in.


