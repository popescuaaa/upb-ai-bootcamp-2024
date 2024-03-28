# Optional assignment

``` Note ```

Check the code starter.

## Customer Lifetime Value prediction

We invest in customers (acquisition costs, offline ads, promotions, discounts & etc.) to generate revenue and be profitable. Naturally, these actions make some customers super valuable in terms of lifetime value but there are always some customers who pull down the profitability. We need to identify these behavior patterns, segment customers and act accordingly. Calculating Lifetime Value is the easy part. First we need to select a time window. It can be anything like 3, 6, 12, 24 months. By the equation below, we can have Lifetime Value for each customer in that specific time window:

``` Lifetime Value: Total Gross Revenue - Total Cost ```

This equation now gives us the historical lifetime value. If we see some customers having very high negative lifetime value historically, it could be too late to take an action.

You are going to build a simple machine learning model that predicts our customers lifetime value. There is no restriction for you on this, you can go for very simple methods or even complicated ones; the most important is to present you work and we will evaluate it according to the points specified bellow.

#### Lifetime Value Prediction
1. Define an appropriate time frame for Customer Lifetime Value calculation
2. Identify the features we are going to use to predict future and create them
3. Calculate lifetime value (LTV) for training the machine learning model
4. Build and run the machine learning model
5. Check if the model is useful

#### How to decide the timeframe

Deciding the time frame really depends on your industry, business model, strategy and more. For some industries, 1 year is a very long period while for the others it is very short. In our example, we will go ahead with 6 months.


### Dataset

You will be using a custom dataset gathered from Kaggle. 

Structure:

```bash

InvoiceNo
StockCode,
Description,
Quantity,
InvoiceDate,
UnitPrice,
CustomerID,
Country

```
#### Tips and tricks

- Explore the dataset for the assignment
- Explain in comments and Markdown blocks what you have discovered
- If you want to add / remove features from the dataset is up to you!
- Use existing models for this: scikit-learn, xgboost or any other thing you want.
- Try to use plots as much as you can.

Note: the mark is not dependent on accuracy scores or any other metrics. We evaluate the work based on the points bellow and the presentation.

#### Points to be covered for full grade

- [ ] A correct modeling of the problem 
- [ ] Data exploration 
- [ ] Clean code
- [ ] Comments

#### Note

Keep in mind that you have to present your work at the end and the presentation should be 5-6 minutes.


#### Examples:

Link: https://www.kaggle.com/datasets/sergeymedvedev/customer_segmentation/code